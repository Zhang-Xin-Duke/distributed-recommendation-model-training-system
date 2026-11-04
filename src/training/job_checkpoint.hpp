#pragma once
#include <nlohmann/json.hpp>
namespace eps::training {
nlohmann::json JobCheckpoint(const nlohmann::json& config, const std::string& action);
}
