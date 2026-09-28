#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace splash::ops {

// A table of installed operator choices, each a workload and its
// configuration: sorted by workload, one choice per workload.
template <class Choice>
void sortUniqueChoices(std::vector<Choice> &choices) {
  std::sort(choices.begin(), choices.end(),
            [](const Choice &a, const Choice &b) { return a.workload < b.workload; });
  if (std::adjacent_find(choices.begin(), choices.end(), [](const Choice &a, const Choice &b) {
        return a.workload == b.workload;
      }) != choices.end())
    throw std::invalid_argument("duplicate operator choice");
}

// The configuration a sorted table holds for `workload`, or `baseline`.
template <class Choice, class Workload, class Configuration>
Configuration chosenConfiguration(const std::vector<Choice> &choices, const Workload &workload,
                                  Configuration baseline) {
  const auto found = std::lower_bound(choices.begin(), choices.end(), workload,
                                      [](const Choice &choice, const Workload &key) { return choice.workload < key; });
  return found != choices.end() && found->workload == workload ? found->configuration : baseline;
}

} // namespace splash::ops
