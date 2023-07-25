
add_library(shewchuk STATIC shewchuk.c shewchuk.h)
# Use set_target_properties to clear existing compile options (especially -Werr)
set_target_properties(shewchuk PROPERTIES COMPILE_OPTIONS "")
target_compile_options(shewchuk PRIVATE -ffp-contract=on -w)
