#pragma once
#include <string>
#include <vector>
struct mtl_number_rewrite {
    std::string source;
    std::string target;
};
struct mtl_number_result {
    std::string text;
    std::vector<mtl_number_rewrite> rewrites;
    std::string provider;
};
struct mtl_numbers {
    mtl_number_result verbalize_numbers(const std::string & text, const std::string & language_id) const;
};
