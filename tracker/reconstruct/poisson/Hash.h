#ifndef HASH_INCLUDED
#define HASH_INCLUDED
#ifdef WIN32
#include <hash_map>
using stdext::hash_map;
#else // !WIN32
//hash_map is deprecated with new GCC, see webpage:
// http://fgda.pl/post/7/gcc-hash-map-vs-unordered-map
#define GCC_VERSION (__GNUC__ * 10000 \
    + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#if GCC_VERSION >= 40300
#include <tr1/unordered_map>
#define hash_map std::tr1::unordered_map
#else
#include <ext/hash_map>
#define hash_map __gnu_cxx::hash_map

using namespace __gnu_cxx;

namespace __gnu_cxx
{
  template<> struct hash<long long> {
    size_t operator()(long long __x) const { return __x; }
  };
  template<> struct hash<const long long> {
    size_t operator()(const long long __x) const { return __x; }
  };


  template<> struct hash<unsigned long long> {
    size_t operator()(unsigned long long __x) const { return __x; }
  };
  template<> struct hash<const unsigned long long> {
    size_t operator()(const unsigned long long __x) const { return __x; }
  };
}

#endif //GCC_VERSION

#endif // WIN32
#endif // HASH_INCLUDED

