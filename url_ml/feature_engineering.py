import ipaddress
import math
import re
from collections import Counter
from typing import Dict, List, Tuple
from urllib.parse import parse_qsl, unquote, urlparse

NUMERIC_FEATURES: List[str] = [
    "URLLength",
    "HostnameLength",
    "PathLength",
    "QueryLength",
    "NoOfDots",
    "NoOfHyphens",
    "NoOfDigits",
    "NoOfSlashes",
    "NoOfQueryParams",
    "DigitRatio",
    "SpecialCharRatio",
    "NoOfObfuscatedChar",
    "HostEntropy",
    "PathEntropy",
    "HasAtSymbol",
    "IsIPHost",
    "SubdomainDepth",
    "IsSuspiciousTLD",
    "IsPunycode",
    "HasHTTPS",
    "HasNonStandardPort",
    "NoOfSuspiciousKeywords",
    "HasBrandInSubdomainOrPath",
    "IsShortener",
    "HasEmbeddedURL",
]

SUSPICIOUS_TLDS = {
    "top", "xyz", "club", "work", "online", "site", "info", "loan", "bid", "cam",
    "vip", "tokyo", "fit", "party", "live", "buzz", "icu", "rest", "ru", "cn",
    "tk", "ml", "ga", "cf", "gq", "kim", "wang", "link", "click", "gdn"
}

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "account", "update", "secure",
    "confirm", "password", "wallet", "bank", "security", "authenticate"
]
KEYWORD_PATTERN = re.compile(r"(?:" + "|".join(SUSPICIOUS_KEYWORDS) + r")", re.IGNORECASE)

TARGET_BRANDS = [
    "paypal", "google", "microsoft", "apple", "amazon", "netflix",
    "facebook", "instagram", "binance", "chase", "wellsfargo", "coinbase",
    "telegram", "whatsapp", "roblox", "steam", "dhl", "fedex", "usps"
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "qr.me", "qrco.de", "me-qr.com", "shorturl.at",
    "cutt.ly", "linktr.ee", "rb.gy", "v.gd", "clck.ru", "s.id"
}

SPECIAL_CHAR_SET = set("@~_%=\&?!*;,+$+")
OBFUSCATION_PATTERN = re.compile(r"%[0-9A-Fa-f]{2}")
EMBEDDED_PATTERN = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)


def normalize_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        return ""
    if "://" not in url:
        return f"http://{url}"
    return url


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def extract_url_features(raw_url: str) -> Dict[str, float]:
    url = (raw_url or "").strip()
    has_http_prefix = "://" in url
    url_parsed_str = url if has_http_prefix else f"http://{url}"

    try:
        parsed = urlparse(url_parsed_str)
    except Exception:
        parsed = urlparse("http://invalid-url.com")

    raw_host = (parsed.netloc or "").split("@")[-1]
    if ":" in raw_host:
        host = raw_host.split(":")[0].lower().strip(".")
        port = raw_host.split(":")[1]
    else:
        host = raw_host.lower().strip(".")
        port = None

    path = parsed.path or ""
    query = parsed.query or ""

    url_len = len(url)
    no_of_digits = sum(c.isdigit() for c in url)
    special_cnt = sum(c in SPECIAL_CHAR_SET for c in url)

    try:
        ipaddress.ip_address(host)
        is_ip_host = 1
    except ValueError:
        is_ip_host = 0

    parts = [p for p in host.split(".") if p]
    if len(parts) >= 2:
        tld = parts[-1]
        registered_domain = ".".join(parts[-2:])
        subdomain_depth = max(0, len(parts) - 2)
    elif len(parts) == 1:
        tld = parts[0]
        registered_domain = parts[0]
        subdomain_depth = 0
    else:
        tld = ""
        registered_domain = ""
        subdomain_depth = 0

    outside_reg = (".".join(parts[:-2]) if len(parts) > 2 else "") + "/" + path
    decoded = unquote(f"{path}?{query}")

    try:
        no_of_query_params = len(parse_qsl(query, keep_blank_values=True))
    except Exception:
        no_of_query_params = 0

    try:
        has_non_standard_port = 1 if (port and int(port) not in (80, 443)) else 0
    except ValueError:
        has_non_standard_port = 0

    return {
        "url": raw_url,
        "URLLength": url_len,
        "HostnameLength": len(host),
        "PathLength": len(path),
        "QueryLength": len(query),
        "NoOfDots": url.count("."),
        "NoOfHyphens": url.count("-"),
        "NoOfDigits": no_of_digits,
        "NoOfSlashes": path.count("/"),
        "NoOfQueryParams": no_of_query_params,
        "DigitRatio": (no_of_digits / url_len) if url_len else 0.0,
        "SpecialCharRatio": (special_cnt / url_len) if url_len else 0.0,
        "NoOfObfuscatedChar": len(OBFUSCATION_PATTERN.findall(url)),
        "HostEntropy": _shannon_entropy(host),
        "PathEntropy": _shannon_entropy(f"{path}?{query}"),
        "HasAtSymbol": 1 if ("@" in (parsed.netloc or "") or "@" in path) else 0,
        "IsIPHost": is_ip_host,
        "SubdomainDepth": subdomain_depth,
        "IsSuspiciousTLD": 1 if tld in SUSPICIOUS_TLDS else 0,
        "IsPunycode": 1 if (any(p.startswith("xn--") for p in parts) or any(ord(c) > 127 for c in host)) else 0,
        "HasHTTPS": 1 if (parsed.scheme.lower() == "https" and has_http_prefix) else 0,
        "HasNonStandardPort": has_non_standard_port,
        "NoOfSuspiciousKeywords": len(KEYWORD_PATTERN.findall(url)),
        "HasBrandInSubdomainOrPath": 1 if any(b in outside_reg for b in TARGET_BRANDS) else 0,
        "IsShortener": 1 if (host in SHORTENERS or registered_domain in SHORTENERS) else 0,
        "HasEmbeddedURL": 1 if EMBEDDED_PATTERN.search(decoded) else 0,
        # Backward-compatible convenience aliases for risk policies and heuristics
        "dom": host,
        "tld": tld,
        "is_https": 1 if (parsed.scheme.lower() == "https" and has_http_prefix) else 0,
        "subdom_cnt": subdomain_depth,
        "entropy": _shannon_entropy(host),
        "path_len": len(path),
        "query_len": len(query),
        "is_ip": is_ip_host,
        "spec_ratio": (special_cnt / url_len) if url_len else 0.0,
        "digit_ratio": (no_of_digits / url_len) if url_len else 0.0,
        "url_len": url_len,
    }