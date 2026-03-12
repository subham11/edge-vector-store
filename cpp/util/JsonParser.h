#pragma once
// ──────────────────────────────────────────────────────────────
//  Minimal JSON helpers — shared across EdgeStore & PackFormat
//
//  For production, swap with simdjson / rapidjson.  These helpers
//  parse only the flat shapes we use.
// ──────────────────────────────────────────────────────────────
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace evs {
namespace json {

// ── lookup helpers ──────────────────────────────────────────

inline std::string getString(const std::string& key,
                             const std::map<std::string, std::string>& kvs) {
    auto it = kvs.find(key);
    return it != kvs.end() ? it->second : "";
}

inline int getInt(const std::string& key,
                  const std::map<std::string, std::string>& kvs,
                  int def = 0) {
    auto it = kvs.find(key);
    return it != kvs.end() ? std::atoi(it->second.c_str()) : def;
}

// ── tokeniser primitives ────────────────────────────────────

inline size_t skipWs(const std::string& s, size_t i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' ||
                            s[i] == '\r' || s[i] == '\t'))
        ++i;
    return i;
}

inline std::string extractStr(const std::string& s, size_t& i) {
    if (i >= s.size() || s[i] != '"') return "";
    ++i;
    std::string result;
    result.reserve(64);
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) {
            ++i;
            switch (s[i]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case '/':  result += '/';  break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += s[i]; break;
            }
        } else {
            result += s[i];
        }
        ++i;
    }
    if (i < s.size()) ++i;
    return result;
}

inline std::string extractToken(const std::string& s, size_t& i) {
    std::string result;
    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' &&
           s[i] != ' ' && s[i] != '\n' && s[i] != '\r' && s[i] != '\t') {
        result += s[i++];
    }
    return result;
}

// ── object / array parsers ──────────────────────────────────

inline std::map<std::string, std::string> parseFlat(const std::string& s) {
    std::map<std::string, std::string> kv;
    size_t i = skipWs(s, 0);
    if (i >= s.size() || s[i] != '{') return kv;
    ++i;

    while (true) {
        i = skipWs(s, i);
        if (i >= s.size() || s[i] == '}') break;

        if (s[i] != '"') { ++i; continue; }
        std::string key = extractStr(s, i);

        i = skipWs(s, i);
        if (i >= s.size() || s[i] != ':') break;
        ++i;
        i = skipWs(s, i);

        std::string value;
        if (s[i] == '"') {
            value = extractStr(s, i);
        } else if (s[i] == '[' || s[i] == '{') {
            int depth = 1;
            char open = s[i], close = (open == '[') ? ']' : '}';
            size_t start = i;
            ++i;
            while (i < s.size() && depth > 0) {
                if (s[i] == open) ++depth;
                else if (s[i] == close) --depth;
                else if (s[i] == '"') { extractStr(s, i); continue; }
                ++i;
            }
            value = s.substr(start, i - start);
        } else {
            value = extractToken(s, i);
        }

        kv[key] = value;

        i = skipWs(s, i);
        if (i < s.size() && s[i] == ',') ++i;
    }
    return kv;
}

inline std::vector<float> parseFloatArray(const std::string& s) {
    std::vector<float> v;
    size_t i = skipWs(s, 0);
    if (i >= s.size() || s[i] != '[') return v;
    ++i;

    size_t commaCount = 0;
    for (size_t j = i; j < s.size() && s[j] != ']'; ++j) {
        if (s[j] == ',') ++commaCount;
    }
    v.reserve(commaCount + 1);

    const char* data = s.c_str();
    while (i < s.size()) {
        while (i < s.size() && (data[i] == ' ' || data[i] == ',' ||
               data[i] == '\n' || data[i] == '\r' || data[i] == '\t'))
            ++i;
        if (i >= s.size() || data[i] == ']') break;

        char* end = nullptr;
        float val = std::strtof(data + i, &end);
        if (end == data + i) break;
        v.push_back(val);
        i = static_cast<size_t>(end - data);
    }
    return v;
}

inline std::vector<std::string> parseStringArray(const std::string& s) {
    std::vector<std::string> v;
    size_t i = skipWs(s, 0);
    if (i >= s.size() || s[i] != '[') return v;
    ++i;
    while (true) {
        i = skipWs(s, i);
        if (i >= s.size() || s[i] == ']') break;
        if (s[i] == '"') {
            v.push_back(extractStr(s, i));
        } else {
            ++i;
        }
        i = skipWs(s, i);
        if (i < s.size() && s[i] == ',') ++i;
    }
    return v;
}

inline std::vector<std::map<std::string, std::string>> parseObjectArray(
    const std::string& s) {
    std::vector<std::map<std::string, std::string>> arr;
    size_t i = skipWs(s, 0);
    if (i >= s.size() || s[i] != '[') return arr;
    ++i;
    while (true) {
        i = skipWs(s, i);
        if (i >= s.size() || s[i] == ']') break;
        if (s[i] == '{') {
            int depth = 1;
            size_t start = i;
            ++i;
            while (i < s.size() && depth > 0) {
                if (s[i] == '{') ++depth;
                else if (s[i] == '}') --depth;
                else if (s[i] == '"') { extractStr(s, i); continue; }
                ++i;
            }
            arr.push_back(parseFlat(s.substr(start, i - start)));
        } else {
            ++i;
        }
        i = skipWs(s, i);
        if (i < s.size() && s[i] == ',') ++i;
    }
    return arr;
}

inline std::string toJsonString(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    out += '"';
    return out;
}

} // namespace json
} // namespace evs
