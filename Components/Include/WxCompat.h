// Minimal wxWidgets-compatible surface for the headless VideoSubFinder core.
// Keep this deliberately small: it exists only to avoid rewriting the legacy
// algorithm while the wxWidgets GUI continues to use the real library.
#pragma once

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#define wxT(value) value

class wxString : public std::string
{
public:
    using std::string::string;
    wxString() = default;
    wxString(const std::string& value) : std::string(value) {}
    wxString(std::string&& value) : std::string(std::move(value)) {}

    const wxString& ToUTF8() const { return *this; }

    template <typename... Args>
    static wxString Format(const char* format, Args&&... args)
    {
        return format_string(format, format_arg(std::forward<Args>(args))...);
    }

    template <typename... Args>
    void Printf(const char* format, Args&&... args)
    {
        *this = Format(format, std::forward<Args>(args)...);
    }

private:
    static const char* format_arg(const wxString& value) { return value.c_str(); }
    static const char* format_arg(const std::string& value) { return value.c_str(); }
    template <typename T>
    static T&& format_arg(T&& value) { return std::forward<T>(value); }

    template <typename... Args>
    static wxString format_string(const char* format, Args... args)
    {
        const int size = std::snprintf(nullptr, 0, format, args...);
        if (size <= 0) return wxString();
        wxString result(static_cast<size_t>(size), '\0');
        std::snprintf(result.data(), result.size() + 1, format, args...);
        return result;
    }
};

inline int wxAtoi(const wxString& value)
{
    try { return std::stoi(value); }
    catch (...) { return 0; }
}

class wxArrayString : public std::vector<wxString>
{
public:
    void Add(const wxString& value) { push_back(value); }
};

class wxRegEx
{
public:
    explicit wxRegEx(const wxString& pattern) : regex_(pattern) {}

    bool Matches(const wxString& value)
    {
        return std::regex_search(value, match_, regex_);
    }

    wxString GetMatch(const wxString&, size_t index) const
    {
        return index < match_.size() ? wxString(match_[index].str()) : wxString();
    }

    size_t ReplaceAll(wxString* value, const wxString& replacement) const
    {
        const auto before = *value;
        *value = std::regex_replace(*value, regex_, replacement);
        return before == *value ? 0 : 1;
    }

private:
    std::regex regex_;
    std::smatch match_;
};

class wxFFileOutputStream
{
public:
    wxFFileOutputStream(const wxString& path, const wxString& mode)
        : stream_(path, std::ios::out | (mode.find('a') != std::string::npos ? std::ios::app : std::ios::trunc)) {}
    void Close() { stream_.close(); }
    std::ofstream& stream() { return stream_; }
private:
    std::ofstream stream_;
};

class wxTextOutputStream
{
public:
    explicit wxTextOutputStream(wxFFileOutputStream& output) : stream_(output.stream()) {}
    template <typename T>
    wxTextOutputStream& operator<<(const T& value) { stream_ << value; return *this; }
    void Flush() { stream_.flush(); }
private:
    std::ofstream& stream_;
};

class wxFileInputStream
{
public:
    explicit wxFileInputStream(const wxString& path) : stream_(path, std::ios::binary | std::ios::ate) {}
    size_t GetSize()
    {
        const auto end = stream_.tellg();
        stream_.seekg(0);
        return end < 0 ? 0 : static_cast<size_t>(end);
    }
    void ReadAll(void* output, size_t size) { stream_.read(static_cast<char*>(output), static_cast<std::streamsize>(size)); }
private:
    std::ifstream stream_;
};

class wxFileOutputStream
{
public:
    explicit wxFileOutputStream(const wxString& path) : stream_(path, std::ios::binary) {}
    bool IsOk() const { return stream_.good(); }
    void Write(const void* data, size_t size) { stream_.write(static_cast<const char*>(data), static_cast<std::streamsize>(size)); }
    void WriteAll(const void* data, size_t size) { Write(data, size); }
    void Close() { stream_.close(); }
private:
    std::ofstream stream_;
};

#define wxASSERT_MSG(condition, message) assert((condition) && (message).c_str())
