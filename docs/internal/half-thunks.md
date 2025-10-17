# Half thunks

This uninspired name refers to libraries that aren't fully wrapped/thunked, but only partially.

It is often the case that there's huge libraries that have some expensive functions that are frequently used and are detached from the rest of the library, and lots of functions that are rarely used, less expensive, and are harder to properly thunk. An example of such a library is libc. The C library has tons of functions and most of them we have no interest in thunking because the performance boost this would provide would either be not worthy or non existant. So we developed a mechanism where only select expensive and frequently used functions are thunked and the rest are left as is. Memcpy and memset are good examples in the case of libc.

During initialization, we can find libc.so.6 and replace the assembly in its memcpy function with assembly that will make a host call in our emulator. We then save it as a copy of the library in the `/tmp` directory and subsequent runs can use it directly. Usually the functions we care about have enough size for us to write our custom code.

The rest will be done naturally by the program. The libraries will be loaded, the symbols resolved, and when the program calls the function it will land on our magic code which will make the host call.