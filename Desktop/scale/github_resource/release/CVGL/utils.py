import math
import os
import sys
import random
import errno
import time
import torch
import numpy as np
from datetime import timedelta
import os
import shutil

class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def setup_system(seed, cudnn_benchmark=True, cudnn_deterministic=True) -> None:
    '''
    Set seeds for reproducible training
    '''
    # python
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.buffer = []  # 缓冲区：用于暂存当前行的内容
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'a')  # 使用追加模式

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        # 实时输出到控制台
        self.console.write(msg)
        self.console.flush()

        # 处理消息：逐字符分析
        for char in msg:
            if char == '\r':
                # 遇到回车符：清空缓冲区（丢弃当前行的中间状态）
                self.buffer = []
            elif char == '\n':
                # 关键修改：即使缓冲区为空也写入换行符
                if self.file is not None:
                    line = ''.join(self.buffer)
                    self.file.write(line + '\n')  # 空缓冲区会写入单独换行符
                    self.flush_file()
                self.buffer = []  # 重置缓冲区
            else:
                # 普通字符：加入缓冲区
                self.buffer.append(char)

    def flush(self):
        self.console.flush()
        self.flush_file()

    def flush_file(self):
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        # 关闭前写入缓冲区剩余内容（确保最后一行不丢失）
        if self.buffer:
            if self.file is not None:
                line = ''.join(self.buffer)
                self.file.write(line + '\n')
        if self.file is not None:
            self.file.close()
            self.file = None

    def isatty(self):
        return self.console.isatty()


def sec_to_min(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    seconds_remaining = seconds % 60

    if seconds_remaining < 10:
        seconds_remaining = '0{}'.format(seconds_remaining)

    return '{}:{}'.format(minutes, seconds_remaining)


def sec_to_time(seconds):
    return "{:0>8}".format(str(timedelta(seconds=int(seconds))))


def print_time_stats(t_train_start, t_epoch_start, epochs_remaining, steps_per_epoch):
    elapsed_time = time.time() - t_train_start
    speed_epoch = time.time() - t_epoch_start
    speed_batch = speed_epoch / steps_per_epoch
    eta = speed_epoch * epochs_remaining

    print("Elapsed {}, {} time/epoch, {:.2f} s/batch, remaining {}".format(
        sec_to_time(elapsed_time), sec_to_time(speed_epoch), speed_batch, sec_to_time(eta)))


def save_source_files(save_path, additional_files=None, current_script=None):
    """
    将当前运行脚本及指定文件复制到保存目录

    Args:
        save_path (str): 保存目录路径
        additional_files (list, optional): 需要额外复制的文件列表（相对于脚本目录）
        current_script (str, optional): 当前运行脚本路径（默认自动获取）

    Returns:
        list: 成功复制的文件名列表

    ################################ 使用示例 ################################
    save_subfolder_name = f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_{opt.run_tag}'
    save_base_path = f'{opt.save_base_path}/{save_subfolder_name}'
    logger = Logger(os.path.join(save_base_path, 'log.txt'))
    sys.stdout = logger
    sys.stderr = logger

    # ===== 新增调用封装函数 =====
    # 指定需要额外保存的文件（可选）
    additional_files = [
        'model.py',
        'utils/__init__.py',  # 支持子目录
        'config.py',
        'trainer.py'
    ]

    # 调用函数保存文件
    copied_files = save_source_files(
        save_path=save_base_path,
        additional_files=additional_files
    )
    """
    import os
    import shutil
    import sys

    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)

    # 获取当前脚本路径
    if current_script is None:
        current_script = os.path.abspath(sys.argv[0])
    script_dir = os.path.dirname(current_script)

    # 准备文件列表
    files_to_copy = [current_script]
    if additional_files:
        for f in additional_files:
            # 处理可能的子目录路径
            file_path = os.path.normpath(os.path.join(script_dir, f))
            if os.path.exists(file_path):
                files_to_copy.append(file_path)
            else:
                print(f"️  文件未找到: {f} (在目录 {script_dir} 中)")

    # 执行复制
    copied_files = []
    for src in files_to_copy:
        try:
            filename = os.path.basename(src)
            dst = os.path.join(save_path, filename)
            shutil.copy2(src, dst)
            copied_files.append(filename)
        except Exception as e:
            print(f"   复制失败 {os.path.basename(src)}: {str(e)}")

    # 打印保存信息
    print(f"\n{'=' * 50}")
    print(f"已保存源代码至: {os.path.abspath(save_path)}")
    print(f"已保存文件: {', '.join(copied_files)}")
    print(f"{'=' * 50}\n")

    return copied_files


def float_to_32(num):
    return math.ceil(num/32) * 32


def save_used_code(save_path, ignore_patterns=None):
    """
    保存当前运行环境中所有属于项目根目录的源码文件。
    :param save_path: 保存的目标文件夹
    :param ignore_patterns: 不需要保存的文件名列表 (如 'temp_test.py', 'debug.py')
    """
    if ignore_patterns is None:
        ignore_patterns = []

    # 获取项目根目录 (假设当前运行脚本位于项目内，取其公共前缀或当前工作目录)
    project_root = os.getcwd()

    # 确保目标目录存在
    # src_backup_dir = os.path.join(save_path, 'source_code')
    src_backup_dir = save_path
    mkdir_if_missing(src_backup_dir)

    print(f"Backing up source code to {src_backup_dir}...")

    saved_count = 0
    # 遍历所有加载的模块
    for name, module in sys.modules.items():
        # 跳过没有文件的内置模块
        if not hasattr(module, '__file__') or module.__file__ is None:
            continue

        file_path = os.path.abspath(module.__file__)

        # 1. 检查文件是否在项目目录下 (排除 site-packages, anaconda 等库文件)
        if not file_path.startswith(project_root):
            continue

        # 2. 检查是否在忽略列表中
        file_name = os.path.basename(file_path)
        if any(pat in file_name for pat in ignore_patterns):
            continue

        # 3. 检查扩展名，只保存源码
        if not file_name.endswith('.py'):
            continue

        # 计算相对路径，保持目录结构
        rel_path = os.path.relpath(file_path, project_root)
        dest_path = os.path.join(src_backup_dir, rel_path)

        # 创建子目录并复制
        mkdir_if_missing(os.path.dirname(dest_path))
        shutil.copy(file_path, dest_path)
        saved_count += 1

    # 额外保存主运行脚本 (因为 sys.modules['__main__'] 的路径有时不准确)
    main_script = os.path.abspath(sys.argv[0])
    if main_script.startswith(project_root):
        main_name = os.path.basename(main_script)
        if main_name not in ignore_patterns:
            rel_path = os.path.relpath(main_script, project_root)
            dest_path = os.path.join(src_backup_dir, rel_path)
            mkdir_if_missing(os.path.dirname(dest_path))
            shutil.copy(main_script, dest_path)
            # print(f"Saved main script: {rel_path}")

    print(f"Successfully saved {saved_count} source files.")