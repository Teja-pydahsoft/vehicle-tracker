import urllib.parse

def sanitize_rtsp_url(url):
    if not isinstance(url, str) or not url.startswith('rtsp://'):
        return url
    try:
        prefix = 'rtsp://'
        url_stripped = url[len(prefix):]
        if '@' not in url_stripped:
            return url
        parts = url_stripped.rsplit('@', 1)
        if len(parts) != 2:
            return url
        creds, host_part = parts
        if ':' in creds:
            user, password = creds.split(':', 1)
            if '%' not in password:
                encoded_password = urllib.parse.quote(password)
                return f"{prefix}{user}:{encoded_password}@{host_part}"
        return url
    except Exception as e:
        return url

test_urls = [
    ("rtsp://admin:pydah@123@192.168.5.58:554/h264Preview_01_main", "rtsp://admin:pydah%40123@192.168.5.58:554/h264Preview_01_main"),
    ("rtsp://user:pass@host", "rtsp://user:pass@host"),
    ("rtsp://user:p@ssw@rd@host", "rtsp://user:p%40ssw%40rd@host"),
    ("rtsp://host/path", "rtsp://host/path"),
    ("rtsp://user:pass%40word@host", "rtsp://user:pass%40word@host")
]

for input_url, expected in test_urls:
    result = sanitize_rtsp_url(input_url)
    print(f"Input:    {input_url}")
    print(f"Output:   {result}")
    print(f"Expected: {expected}")
    assert result == expected
    print("MATCH ✓")
    print("-" * 20)

print("All tests passed!")
