using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
/// <summary>
/// TCP 服务端（监听端），接收 Python 管线发来的 WHAM / GMR 画面帧，显示在 RawImage 上。
///
/// 用法：
///   1. 将此脚本挂在 Canvas 上，拖入 whamRawImage / gmrRawImage
///   2. 运行场景 → 自动开始监听 127.0.0.1:9876
///   3. Python 端以客户端身份连接并发送帧（协议见 SendFrame 示例）
///
/// 协议（每条消息 8 + N 字节）：
///   [4 字节 streamId, big-endian]  0=WHAM, 1=GMR
///   [4 字节 jpgLength, big-endian]
///   [jpgLength 字节]              JPEG 数据
/// </summary>
public class StreamReceiver : MonoBehaviour
{
    [Header("监听")]
    [SerializeField] private string bindAddress = "127.0.0.1";
    [SerializeField] private int port = 9876;

    [Header("显示目标")]
    [SerializeField] private RawImage whamRawImage;
    [SerializeField] private RawImage gmrRawImage;
    [SerializeField] private Color placeholderColor = new Color(0.15f, 0.15f, 0.15f);

    [Header("调试")]
    [SerializeField] private bool logFrameReceive;

    // ── 内部状态 ──────────────────────────────────────────────────────────
    private TcpListener _listener;
    private Thread _acceptThread;
    private readonly object _lock = new object();
    private volatile bool _running;

    // 待消费帧
    private byte[] _pendingWham;
    private byte[] _pendingGmr;
    private bool _whamDirty;
    private bool _gmrDirty;

    private Texture2D _whamTex;
    private Texture2D _gmrTex;

    /// <summary>当前是否有客户端连着。</summary>
    public bool HasClient
    {
        get { lock (_lock) return _activeClients.Count > 0; }
    }
    private readonly List<TcpClient> _activeClients = new List<TcpClient>();

    // ── Unity 生命周期 ────────────────────────────────────────────────────

    void OnEnable()
    {
        _running = true;
        StartListener();
    }

    void OnDisable()
    {
        _running = false;
        StopListener();
    }

    void Update()
    {
        lock (_lock)
        {
            if (_whamDirty && _pendingWham != null)
            {
                ApplyJpg(ref _whamTex, _pendingWham, whamRawImage);
                _pendingWham = null;
                _whamDirty = false;
            }
            if (_gmrDirty && _pendingGmr != null)
            {
                ApplyJpg(ref _gmrTex, _pendingGmr, gmrRawImage);
                _pendingGmr = null;
                _gmrDirty = false;
            }
        }

        // 清理断开的客户端
        lock (_lock)
            _activeClients.RemoveAll(c => !c.Connected);
    }

    // ── 监听 ──────────────────────────────────────────────────────────────

    void StartListener()
    {
        try
        {
            _listener = new TcpListener(IPAddress.Parse(bindAddress), port);
            _listener.Start();
            Debug.Log($"[StreamRecv] 开始监听 {bindAddress}:{port}");

            _acceptThread = new Thread(AcceptLoop)
            {
                Name = "StreamRecv-Accept",
                IsBackground = true
            };
            _acceptThread.Start();
        }
        catch (Exception e)
        {
            Debug.LogError($"[StreamRecv] 启动监听失败: {e.Message}");
        }
    }

    void StopListener()
    {
        lock (_lock) _activeClients.Clear();
        try { _listener?.Stop(); } catch { }
        _listener = null;
    }

    void AcceptLoop()
    {
        while (_running && _listener != null)
        {
            try
            {
                var client = _listener.AcceptTcpClient();
                client.ReceiveTimeout = 5000;
                client.SendTimeout = 5000;
                lock (_lock) _activeClients.Add(client);

                var t = new Thread(() => ClientRecvLoop(client))
                {
                    Name = "StreamRecv-Client",
                    IsBackground = true
                };
                t.Start();

                Debug.Log($"[StreamRecv] 客户端已连接 ({client.Client.RemoteEndPoint})");
            }
            catch (SocketException)
            {
                if (!_running) break;
                Thread.Sleep(200);
            }
            catch (ObjectDisposedException) { break; }
            catch (Exception e)
            {
                Debug.LogWarning($"[StreamRecv] Accept 异常: {e.Message}");
                Thread.Sleep(200);
            }
        }
    }

    // ── 单客户端接收 ──────────────────────────────────────────────────────

    void ClientRecvLoop(TcpClient client)
    {
        var header = new byte[8];
        NetworkStream stream = null;
        try { stream = client.GetStream(); }
        catch { return; }

        while (_running && client.Connected)
        {
            try
            {
                if (!ReadExact(stream, header, 0, 8))
                    break;

                // 大端解码
                int streamId = (header[0] << 24) | (header[1] << 16) | (header[2] << 8) | header[3];
                int jpgLen   = (header[4] << 24) | (header[5] << 16) | (header[6] << 8) | header[7];

                if (jpgLen <= 0 || jpgLen > 10 * 1024 * 1024)
                    break;

                var jpg = new byte[jpgLen];
                if (!ReadExact(stream, jpg, 0, jpgLen))
                    break;

                lock (_lock)
                {
                    if (streamId == 0) { _pendingWham = jpg; _whamDirty = true; }
                    else if (streamId == 1) { _pendingGmr = jpg; _gmrDirty = true; }
                }

                if (logFrameReceive)
                    Debug.Log($"[StreamRecv] #{streamId} {jpgLen}B");
            }
            catch (IOException) { break; }
            catch (SocketException) { break; }
            catch (Exception e)
            {
                Debug.LogWarning($"[StreamRecv] 客户端读取异常: {e.Message}");
                break;
            }
        }

        // 清理
        lock (_lock) _activeClients.Remove(client);
        try { stream?.Close(); } catch { }
        try { client.Close(); } catch { }
        Debug.Log("[StreamRecv] 客户端已断开");
    }

    // ── 辅助 ──────────────────────────────────────────────────────────────

    static bool ReadExact(NetworkStream s, byte[] buf, int offset, int count)
    {
        while (count > 0)
        {
            int n = s.Read(buf, offset, count);
            if (n <= 0) return false;
            offset += n;
            count  -= n;
        }
        return true;
    }

    static void ApplyJpg(ref Texture2D tex, byte[] jpg, RawImage image)
    {
        if (image == null) return;
        if (tex == null)
            tex = new Texture2D(2, 2, TextureFormat.RGB24, false);

        if (tex.LoadImage(jpg))
            image.texture = tex;
    }

    public void ShowPlaceholder()
    {
        if (whamRawImage != null) { whamRawImage.texture = null; whamRawImage.color = placeholderColor; }
        if (gmrRawImage != null) { gmrRawImage.texture = null; gmrRawImage.color = placeholderColor; }
    }
}
