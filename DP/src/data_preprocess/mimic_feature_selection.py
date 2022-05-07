import os
import re
from util import *


def main():
    unpreprocessed_path = os.path.abspath('../../resource/preprocessed_data/mimic_unpreprocessed.csv')
    save_path = os.path.abspath('../../resource/preprocessed_data/mimic_after_variable_selection.csv')
    feature_discard_threshold = 0.3

    data_dict = general_read_data_to_dict(unpreprocessed_path, skip_extra_line=0)
    print('un preprocessed data size: {}, variable num: {}'
          .format(calculate_visit_count(data_dict), calculate_variable_number(data_dict)))
    data_dict = discard_non_numeric_value(data_dict)
    print('discard non numerical value: {}, variable num: {}'
          .format(calculate_visit_count(data_dict), calculate_variable_number(data_dict)))
    data_dict = delete_feature_missing_too_much(data_dict, feature_discard_threshold)
    print('discard feature with significant missing: {}, variable num: {}'
          .format(calculate_visit_count(data_dict), calculate_variable_number(data_dict)))
    write_data_dict_to_csv(data_dict, save_path, missing_rate=True, median=True)
    print('accomplish')


def delete_feature_missing_too_much(data_dict, feature_delete_missing_rate):
    missing_dict = dict()
    total_count = 0
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            total_count += 1
            for item in data_dict[patient_id][visit_id]:
                missing_dict[item] = 0

    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            for item in data_dict[patient_id][visit_id]:
                value = data_dict[patient_id][visit_id][item]
                if value == -1 or value.__contains__('-1'):
                    missing_dict[item] += 1

    discard_feature_set = set()
    for item in missing_dict:
        missing_count = missing_dict[item]
        if missing_count / total_count > feature_delete_missing_rate:
            discard_feature_set.add(item)

    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            for item in discard_feature_set:
                data_dict[patient_id][visit_id].pop(item)
    return data_dict


def discard_preset_feature(data_dict, preset_discard_feature_set):
    for item in preset_discard_feature_set:
        for patient_id in data_dict:
            for visit_id in data_dict[patient_id]:
                if data_dict[patient_id][visit_id].__contains__(item):
                    data_dict[patient_id][visit_id].pop(item)
                else:
                    print('Error: {} is not in dataset'.format(item))
    return data_dict


def composite_feature(data_dict, feature_composite):
    new_dict = dict()
    for patient_id in data_dict:
        new_dict[patient_id] = dict()
        for visit_id in data_dict[patient_id]:
            new_dict[patient_id][visit_id] = dict()
            for feature in data_dict[patient_id][visit_id]:
                if feature_composite.__contains__(feature):
                    mapped_feature_name = feature_composite[feature]
                else:
                    mapped_feature_name = feature

                if not new_dict[patient_id][visit_id].__contains__(mapped_feature_name):
                    new_dict[patient_id][visit_id][mapped_feature_name] = \
                        data_dict[patient_id][visit_id][feature]
                else:
                    # 按照当前的设计，可以整合的变量必须是0-1变量，因此一定可以数值化，如果不能数值化代表哪里出错了
                    origin_value = int(new_dict[patient_id][visit_id][mapped_feature_name])
                    if origin_value == '0' or origin_value == 0:
                        new_dict[patient_id][visit_id][mapped_feature_name] = \
                            data_dict[patient_id][visit_id][feature]
    return new_dict


def discard_non_numeric_value(data_dict):
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            for item in data_dict[patient_id][visit_id]:
                value = str(data_dict[patient_id][visit_id][item])
                value_list = re.findall('[-+]?[\d]+(?:,\d\d\d)*[.]?\d*(?:[eE][-+]?\d+)?', value)
                if len(value_list) == 0:
                    data_dict[patient_id][visit_id][item] = '-1'
                else:
                    data_dict[patient_id][visit_id][item] = str(value_list[0])
    return data_dict


if __name__ == '__main__':
    main()
