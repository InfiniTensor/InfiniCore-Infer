local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

target("infinicore_infer")
    set_kind("shared")

    add_includedirs("include", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("src/models/*.cpp")
    add_files("src/models/*/*.cpp")
    add_files("src/tensor/*.cpp")
    add_files("src/allocator/*.cpp")
    add_files("src/dataloader/*.cpp")
    add_files("src/cache_manager/*.cpp")
    add_includedirs("include")

    set_installdir(INFINI_ROOT)
    add_installfiles("include/infinicore_infer.h", {prefixdir = "include"})
    add_installfiles("include/infinicore_infer/models/*.h", {prefixdir = "include/infinicore_infer/models"})
target_end()
