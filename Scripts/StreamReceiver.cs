using System;
using System.Collections.Concurrent;
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
    [SerializeField] private Color placeholderColor = new Color(0.02f, 0.04f, 0.06f, 0.22f);

    [Header("Aspect")]
    [SerializeField] private bool preserveIncomingAspect = true;
    [SerializeField] private bool resizeRawImageRect = true;
    [SerializeField] private float maxTextureApplyFps = 30f;

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
    private float _nextWhamApplyTime;
    private float _nextGmrApplyTime;
    private long _whamReceivedFrames;
    private long _gmrReceivedFrames;
    private long _whamShownFrames;
    private long _gmrShownFrames;
    private long _lastWhamReceivedSample;
    private long _lastGmrReceivedSample;
    private long _lastWhamShownSample;
    private long _lastGmrShownSample;
    private float _lastFpsSampleTime = -1f;
    private float _whamReceiveFps;
    private float _gmrReceiveFps;
    private float _whamShownFps;
    private float _gmrShownFps;
    private float _lastWhamFrameTime = -1f;
    private float _lastGmrFrameTime = -1f;
    private readonly Dictionary<RawImage, Vector2> _initialImageSizes = new Dictionary<RawImage, Vector2>();
    private readonly ConcurrentQueue<QueuedStreamLog> _logQueue = new ConcurrentQueue<QueuedStreamLog>();

    public void SetHudVideoDisplayOptions(bool hudManagedAspectFit, Color placeholder)
    {
        resizeRawImageRect = !hudManagedAspectFit;
        preserveIncomingAspect = true;
        placeholderColor = placeholder;
    }

    private enum QueuedStreamLogType
    {
        Log,
        Warning,
        Error
    }

    private struct QueuedStreamLog
    {
        public readonly string Message;
        public readonly QueuedStreamLogType Type;

        public QueuedStreamLog(string message, QueuedStreamLogType type)
        {
            Message = message;
            Type = type;
        }
    }

    void Awake()
    {
        CacheInitialRawImageSize(whamRawImage);
        CacheInitialRawImageSize(gmrRawImage);
        ShowPlaceholder();
    }

    /// <summary>当前是否有客户端连着。</summary>
    public bool HasClient
    {
        get { lock (_lock) return _activeClients.Count > 0; }
    }
    private readonly List<TcpClient> _activeClients = new List<TcpClient>();

    public bool IsListening => _listener != null;
    public string BindAddress => bindAddress;
    public int Port => port;

    public static StreamReceiver EnsureReceiverHost()
    {
        StreamReceiver receiver = FindObjectOfType<StreamReceiver>(true);
        if (receiver == null || !receiver.gameObject.activeInHierarchy)
        {
            GameObject host = GameObject.Find("StreamReceiverHost");
            if (host == null)
            {
                host = new GameObject("StreamReceiverHost");
            }

            receiver = host.GetComponent<StreamReceiver>();
            if (receiver == null)
            {
                receiver = host.AddComponent<StreamReceiver>();
            }
        }

        receiver.EnsureListening();
        return receiver;
    }

    public bool EnsureListening()
    {
        if (!gameObject.activeSelf)
        {
            gameObject.SetActive(true);
        }

        if (!enabled)
        {
            enabled = true;
        }

        _running = true;
        if (_listener == null)
        {
            StartListener();
        }

        return _listener != null;
    }

    public void ConfigureTargets(RawImage whamTarget, RawImage gmrTarget)
    {
        whamRawImage = whamTarget;
        gmrRawImage = gmrTarget;
        CacheInitialRawImageSize(whamRawImage);
        CacheInitialRawImageSize(gmrRawImage);
        ApplyExistingTextureOrPlaceholder(_whamTex, whamRawImage);
        ApplyExistingTextureOrPlaceholder(_gmrTex, gmrRawImage);
    }

    public void ClearStream(int streamId)
    {
        if (streamId == 0)
        {
            ClearStream(ref _whamTex, whamRawImage, isWham: true);
        }
        else if (streamId == 1)
        {
            ClearStream(ref _gmrTex, gmrRawImage, isWham: false);
        }
    }

    public void ClearAllStreams()
    {
        ClearStream(0);
        ClearStream(1);
    }

    public string GetStatusLine(string label, int streamId)
    {
        RawImage target = streamId == 0 ? whamRawImage : gmrRawImage;
        Texture texture = target != null ? target.texture : null;
        long received;
        long shown;
        float receiveFps;
        float shownFps;
        float lastTime;
        bool hasClient;
        lock (_lock)
        {
            received = streamId == 0 ? _whamReceivedFrames : _gmrReceivedFrames;
            shown = streamId == 0 ? _whamShownFrames : _gmrShownFrames;
            receiveFps = streamId == 0 ? _whamReceiveFps : _gmrReceiveFps;
            shownFps = streamId == 0 ? _whamShownFps : _gmrShownFps;
            lastTime = streamId == 0 ? _lastWhamFrameTime : _lastGmrFrameTime;
            hasClient = _activeClients.Count > 0;
        }
        string client = hasClient ? "client: connected" : "client: waiting";
        string last = lastTime >= 0f ? $"{Time.unscaledTime - lastTime:F1}s ago" : "never";
        string tex = texture != null ? $"{texture.width}x{texture.height}" : "none";
        string rect = target != null ? $"{Mathf.RoundToInt(target.rectTransform.rect.width)}x{Mathf.RoundToInt(target.rectTransform.rect.height)}" : "missing";
        return $"{label} | {bindAddress}:{port} | {(IsListening ? "listening" : "closed")} | {client} | rx(TCP) {received} @ {receiveFps:F1} fps | shown(UI) {shown} @ {shownFps:F1} fps | last {last} | tex {tex} | target {rect}";
    }

    // ── Unity 生命周期 ────────────────────────────────────────────────────

    void OnEnable()
    {
        _running = true;
        ShowPlaceholder();
        StartListener();
    }

    void OnDisable()
    {
        _running = false;
        StopListener();
    }

    void Update()
    {
        FlushLogQueue();

        byte[] whamToApply = null;
        byte[] gmrToApply = null;
        float now = Time.unscaledTime;
        float interval = maxTextureApplyFps > 0.0f ? 1.0f / maxTextureApplyFps : 0.0f;

        lock (_lock)
        {
            if (_whamDirty && _pendingWham != null && now >= _nextWhamApplyTime)
            {
                whamToApply = _pendingWham;
                _pendingWham = null;
                _whamDirty = false;
                _nextWhamApplyTime = now + interval;
            }
            if (_gmrDirty && _pendingGmr != null && now >= _nextGmrApplyTime)
            {
                gmrToApply = _pendingGmr;
                _pendingGmr = null;
                _gmrDirty = false;
                _nextGmrApplyTime = now + interval;
            }
        }

        if (whamToApply != null)
        {
            ApplyJpg(ref _whamTex, whamToApply, whamRawImage);
        }
        if (gmrToApply != null)
        {
            ApplyJpg(ref _gmrTex, gmrToApply, gmrRawImage);
        }

        lock (_lock)
        {
            _activeClients.RemoveAll(c => !c.Connected);
        }

        UpdateRollingFps(now);
    }

    private void UpdateRollingFps(float now)
    {
        lock (_lock)
        {
            if (_lastFpsSampleTime < 0f)
            {
                _lastFpsSampleTime = now;
                _lastWhamReceivedSample = _whamReceivedFrames;
                _lastGmrReceivedSample = _gmrReceivedFrames;
                _lastWhamShownSample = _whamShownFrames;
                _lastGmrShownSample = _gmrShownFrames;
                return;
            }

            float dt = now - _lastFpsSampleTime;
            if (dt < 1f)
            {
                return;
            }

            _whamReceiveFps = (_whamReceivedFrames - _lastWhamReceivedSample) / dt;
            _gmrReceiveFps = (_gmrReceivedFrames - _lastGmrReceivedSample) / dt;
            _whamShownFps = (_whamShownFrames - _lastWhamShownSample) / dt;
            _gmrShownFps = (_gmrShownFrames - _lastGmrShownSample) / dt;

            _lastFpsSampleTime = now;
            _lastWhamReceivedSample = _whamReceivedFrames;
            _lastGmrReceivedSample = _gmrReceivedFrames;
            _lastWhamShownSample = _whamShownFrames;
            _lastGmrShownSample = _gmrShownFrames;
        }
    }

    // ── 监听 ──────────────────────────────────────────────────────────────

    void StartListener()
    {
        try
        {
            StopListener();
            _listener = new TcpListener(IPAddress.Parse(bindAddress), port);
            _listener.Server.NoDelay = true;
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
            try { _listener?.Stop(); } catch { }
            _listener = null;
            Debug.LogError($"[StreamRecv] 启动监听失败: {e.Message}");
        }
    }

    void StopListener()
    {
        lock (_lock) _activeClients.Clear();
        try { _listener?.Stop(); } catch { }
        _listener = null;
        if (_acceptThread != null)
        {
            try
            {
                if (_acceptThread.IsAlive)
                    _acceptThread.Join(500);
            }
            catch { }
            _acceptThread = null;
        }
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

                EnqueueLog($"[StreamRecv] client connected ({client.Client.RemoteEndPoint})");
            }
            catch (SocketException)
            {
                if (!_running) break;
                Thread.Sleep(200);
            }
            catch (ObjectDisposedException) { break; }
            catch (Exception e)
            {
                EnqueueLog($"[StreamRecv] Accept error: {e.Message}", QueuedStreamLogType.Warning);
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
                    if (streamId == 0) { _pendingWham = jpg; _whamDirty = true; _whamReceivedFrames++; }
                    else if (streamId == 1) { _pendingGmr = jpg; _gmrDirty = true; _gmrReceivedFrames++; }
                }

                if (logFrameReceive)
                    EnqueueLog($"[StreamRecv] #{streamId} {jpgLen}B");
            }
            catch (IOException) { break; }
            catch (SocketException) { break; }
            catch (Exception e)
            {
                EnqueueLog($"[StreamRecv] client read error: {e.Message}", QueuedStreamLogType.Warning);
                break;
            }
        }

        // 清理
        lock (_lock) _activeClients.Remove(client);
        try { stream?.Close(); } catch { }
        try { client.Close(); } catch { }
        EnqueueLog("[StreamRecv] client disconnected");
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

    void ApplyJpg(ref Texture2D tex, byte[] jpg, RawImage image)
    {
        if (image == null) return;
        if (tex == null)
            tex = new Texture2D(2, 2, TextureFormat.RGB24, false);

        if (tex.LoadImage(jpg))
        {
            image.texture = tex;
            image.color = Color.white;
            FitRawImageToTexture(image, tex.width, tex.height);
            if (image == whamRawImage)
            {
                _whamShownFrames++;
                _lastWhamFrameTime = Time.unscaledTime;
            }
            else if (image == gmrRawImage)
            {
                _gmrShownFrames++;
                _lastGmrFrameTime = Time.unscaledTime;
            }
        }
    }

    private void ApplyExistingTextureOrPlaceholder(Texture2D texture, RawImage image)
    {
        if (image == null)
        {
            return;
        }

        if (texture != null)
        {
            image.texture = texture;
            image.color = Color.white;
            FitRawImageToTexture(image, texture.width, texture.height);
        }
        else
        {
            image.texture = null;
            image.color = placeholderColor;
        }
    }

    private void ClearStream(ref Texture2D texture, RawImage image, bool isWham)
    {
        lock (_lock)
        {
            if (isWham)
            {
                _pendingWham = null;
                _whamDirty = false;
                _whamReceivedFrames = 0;
                _whamShownFrames = 0;
                _lastWhamFrameTime = -1f;
            }
            else
            {
                _pendingGmr = null;
                _gmrDirty = false;
                _gmrReceivedFrames = 0;
                _gmrShownFrames = 0;
                _lastGmrFrameTime = -1f;
            }
        }

        if (texture != null)
        {
            Destroy(texture);
            texture = null;
        }

        if (image != null)
        {
            image.texture = null;
            image.color = placeholderColor;
        }
    }

    private void EnqueueLog(string message, QueuedStreamLogType type = QueuedStreamLogType.Log)
    {
        _logQueue.Enqueue(new QueuedStreamLog(message, type));
    }

    private void FlushLogQueue()
    {
        for (int i = 0; i < 50 && _logQueue.TryDequeue(out QueuedStreamLog item); i++)
        {
            switch (item.Type)
            {
                case QueuedStreamLogType.Warning:
                    Debug.LogWarning(item.Message);
                    break;
                case QueuedStreamLogType.Error:
                    Debug.LogError(item.Message);
                    break;
                default:
                    Debug.Log(item.Message);
                    break;
            }
        }
    }

    void CacheInitialRawImageSize(RawImage image)
    {
        if (image == null || image.rectTransform == null) return;
        if (_initialImageSizes.ContainsKey(image)) return;

        Vector2 size = image.rectTransform.rect.size;
        if (size.x <= 0.0f || size.y <= 0.0f)
        {
            size = image.rectTransform.sizeDelta;
        }
        if (size.x > 0.0f && size.y > 0.0f)
        {
            _initialImageSizes.Add(image, size);
        }
    }

    void FitRawImageToTexture(RawImage image, int width, int height)
    {
        if (!preserveIncomingAspect || !resizeRawImageRect) return;
        if (image == null || image.rectTransform == null || width <= 0 || height <= 0) return;

        CacheInitialRawImageSize(image);
        Vector2 box = ResolveAspectFitBox(image, width, height);

        float frameAspect = width / (float)height;
        float boxAspect = box.x / box.y;
        Vector2 fittedSize = frameAspect >= boxAspect
            ? new Vector2(box.x, box.x / frameAspect)
            : new Vector2(box.y * frameAspect, box.y);

        RectTransform rect = image.rectTransform;
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = Vector2.zero;
        image.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, fittedSize.x);
        image.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, fittedSize.y);
    }

    private Vector2 ResolveAspectFitBox(RawImage image, int width, int height)
    {
        RectTransform rect = image != null ? image.rectTransform : null;
        RectTransform parent = rect != null ? rect.parent as RectTransform : null;
        if (parent != null && parent.rect.width > 1.0f && parent.rect.height > 1.0f)
        {
            return parent.rect.size;
        }

        if (rect != null && rect.rect.width > 1.0f && rect.rect.height > 1.0f)
        {
            return rect.rect.size;
        }

        if (image != null && _initialImageSizes.TryGetValue(image, out Vector2 cached) && cached.x > 1.0f && cached.y > 1.0f)
        {
            return cached;
        }

        return new Vector2(width, height);
    }

    public void ShowPlaceholder()
    {
        if (whamRawImage != null) { whamRawImage.texture = null; whamRawImage.color = placeholderColor; }
        if (gmrRawImage != null) { gmrRawImage.texture = null; gmrRawImage.color = placeholderColor; }
    }
}
