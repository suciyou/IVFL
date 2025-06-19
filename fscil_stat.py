import os
import sys
from alisuretool.Tools import Tools


def read_log(log_file):
    lines = open(log_file, "r").readlines()
    result = {
        "session": os.path.basename(os.path.split(log_file)[0]),
        "log_file": log_file,
        "acc": 0.0,
        "f1": 0.0,
    }
    for one in lines:
        if "epoch: " in one:
            acc = [one for one in one.split("|") if "accuracy" in one][0].split(":")[1].strip()
            f1 = [one for one in one.split("|") if "macro_f1" in one][0].split(":")[1].strip()
            if float(acc) > float(result["acc"]):
                result["acc"] = acc
                result["f1"] = f1
        pass
    return result


def read_result(log_path, is_print=False):
    all_session = os.listdir(log_path)
    total_path = "/root/autodl-tmp/CLIP-TaI/output/total.txt"
    if len(all_session) > 0:
        all_session = sorted(all_session)
        results = []
        for one_session in all_session:
            one_log_file = os.path.join(log_path, one_session, "log.txt")
            result_one = read_log(log_file=one_log_file)
            results.append(result_one)
            pass

        if is_print:
            Tools.print("-" * 100)
            Tools.print(log_path, txt_path=total_path)
            # Tools.print(all_session)
            acc_list = [(result["acc"] if "acc" in result else "0.0") for result in results]
            Tools.print("session |  " +
                        "   |  ".join([result["session"].replace("session", "") for result in results]) +
                        "   |  Avg |  PD  |", txt_path=total_path)
            Tools.print("acc     | " + " | ".join(acc_list) + " | " +
                        f"{sum([float(one) for one in acc_list]) / len(acc_list):.2f}" + " | " +
                        f"{(float(acc_list[0]) - float(acc_list[-1])):.2f}" + " |", txt_path=total_path)
            pass
        pass
    else:
        raise Exception("error, no session")
    return results


def calculate_avg(result):
    acc_list = [float(res["acc"]) for res in result]
    return sum(acc_list) / len(acc_list)


if __name__ == '__main__':
    argv = sys.argv[1:]  # name
    name = argv[0] if len(argv) else None

    total_path = "/root/autodl-tmp/CLIP-TaI/output/total.txt"
    with open(total_path, "w") as f:
        pass

    log_root = "/root/autodl-tmp/CLIP-TaI/output"
    if not os.path.exists(log_root):
        log_root = "/root/autodl-tmp/CLIP-TaI/output"
    print(name)
    if name:
        root_path = os.path.join(log_root, name)
        print('baseline')
        miniImageNet_result = read_result(os.path.join(root_path, "miniImageNet"), is_print=True)
        cifar100_result = read_result(os.path.join(root_path, "cifar100"), is_print=True)
        cub200_result = read_result(os.path.join(root_path, "cub200"), is_print=True)

        mini_avg = calculate_avg(miniImageNet_result)
        cifar_avg = calculate_avg(cifar100_result)
        cub_avg = calculate_avg(cub200_result)

        # 计算总均值
        total_mean = (mini_avg + cifar_avg + cub_avg) / 3

        # 格式化输出
        header = "miniImageNet|    CIFAR100|    CUB200|    Mean"
        values = f"    {mini_avg:.2f}|        {cifar_avg:.2f}|      {cub_avg:.2f}|    {total_mean:.2f}"

        Tools.print("-" * len(header), txt_path=total_path)
        Tools.print(header, txt_path=total_path)
        Tools.print(values, txt_path=total_path)
        Tools.print("-" * len(header))
        pass
    else:
        Tools.print("name error")
        pass
    pass
