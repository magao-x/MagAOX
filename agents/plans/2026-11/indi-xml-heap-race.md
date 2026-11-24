# INDI XML heap race and libcommon thread-safety review

Branch: `jrmales/indi-xml-heap-race`

## Original defect

`IndiXmlParser::createSafeXmlString()` returned `std::string(entityXML(...))`.
lilxml's `entityXML()` keeps its result in one process-global `malloc` buffer
that is reallocated or freed on every call, so concurrent publications from
different threads could free or overwrite each other's escape result
(heap use-after-free / double free). Commit `f2f60f2b` replaced it with a
local escaping loop with identical output (same five entities, stops at an
embedded null).

Verified with a ThreadSanitizer build of `INDI/tests/INDI_test.cpp`: the
pre-fix parser fails the concurrency test and reports races in `entityXML`;
the fixed parser passes.

## Further review of libcommon / liblilxml

ThreadSanitizer on the fixed parser still reported a race in the same publish
path, and review found related latent hazards. All are fixed on this branch:

1. `TimeStamp` used `gmtime()` (shared static `struct tm`) in 11 places,
   including `getFormattedIso8601Str()`, which runs for every published
   message. Concurrent publishers could emit each other's timestamps.
   Replaced with `gmtime_r()` and a local `tm`.
2. `entityXML()` is still reached by lilxml's `sprlXMLEle`/`sprXMLEle`/
   `prXMLEle`/`cloneXMLEle`, i.e. `IndiXmlParser::createXmlString()` for
   parsed messages. The scratch buffer is now `_Thread_local`, keeping the
   documented "use before the next call" contract per thread. Each thread
   that last escaped an entity-bearing string keeps one small buffer until
   exit; MagAO-X threads are long-lived.
3. `IndiXmlParser::createXmlString()` leaked its `malloc` buffer on every call
   for parsed messages. It now prints directly into the returned string.
4. `TimeStamp::local_timegm()` changed `TZ` with `setenv`/`unsetenv`/`tzset`,
   which is unsafe while other threads read the environment. Removed; the
   calendar constructor now uses `timegm()` like `fromFormattedIso8601Str()`.
5. `SystemSocket::getInterfaces()` used `inet_ntoa()` (static buffer).
   Replaced with `inet_ntop()` into a local buffer.

The new parsed-message test exposed two more pre-existing bugs:

6. lilxml set `pcdata_hasent` only for unrecognized entities, so decoded
   `&lt;`, `&amp;`, etc. in element content were printed raw (invalid XML from
   `createXmlString()` and corrupted `cloneXMLEle()` round trips). It is now
   set for every entity.
7. `IndiXmlParser`'s copy constructor and assignment called
   `cloneLilXML()`/`cloneXMLEle()` on a NULL element and segfaulted for any
   parser not mid-parse; a mid-parse copy could not reproduce lilxml's parser
   state anyway. Nothing in the repo copies a parser, so both are now deleted.
   `cloneXMLEle(NULL)` also returns NULL, matching `delXMLEle(NULL)`.

Not changed: `strerror` (glibc returns static table strings for known errors)
and `readdir` (per-directory stream) usages in libcommon.

## Verification

- `INDI_test`: 8 test cases, 59 assertions pass (normal and TSan builds;
  TSan reports nothing).
- Against the old lilxml, the new tests fail 2 cases and TSan reports
  `entityXML` races.
- Clean rebuild of `libcommon`; `indiserver` and `getINDI` link against the
  updated `liblilxml`.
