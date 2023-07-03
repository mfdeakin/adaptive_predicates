
add_library(shewchuk STATIC shewchuk.c shewchuk.h)
target_compile_options(shewchuk PRIVATE "-ffp-contract=on")
