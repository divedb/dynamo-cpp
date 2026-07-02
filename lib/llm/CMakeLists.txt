add_library(dynamo_llm
    src/protocols.cpp
    src/backend.cpp
    src/tokenizer.cpp
    src/router.cpp
)

target_include_directories(dynamo_llm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(dynamo_llm
    PUBLIC
        dynamo_runtime
        nlohmann_json::nlohmann_json
        Folly::folly
)
