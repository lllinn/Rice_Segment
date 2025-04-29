import sys
sys.path.append('./')
import subprocess
import shlex
import os
import sys # Import sys module for exit
from src.utils.email_utils import send_email

# --- 配置 ---
USERNAME = "root"  # 替换成您的服务器用户名
SERVER_IP = "js1.blockelite.cn"    # 替换成您的服务器 IP 地址
REMOTE_BASE_PATH = "/root/test/" # 服务器上数据盘的挂载路径 (确保末尾有 /)
# 如果 rsync 不在 PATH 中，请提供完整路径, 例如: r"C:\Program Files\Git\usr\bin\rsync.exe"
RSYNC_PATH = "rsync.exe"
ssh_command = f"ssh -p 21332"

# 要上传的文件/文件夹列表: (本地路径, 远程子目录或文件名, 是否为目录)
UPLOAD_ITEMS = [
    # --- 修改下面的本地路径为您实际的路径 ---
    (r"E:/Code/RiceLodging/datasets/Meiju1_2_Lingtangkou/abnormal-03.20-7-640-0.1-0.6-0.2-0.2-v6.tar.gz", "Graduation.tar.gz", False),   # 源路径名，放到服务器上的文件名
    # (r"E:/Code/RiceLodging/datasets/Meiju1_2_Lingtangkou/abnormal-03.20-7-640-0.1-0.6-0.2-0.2-v6.tar", "Graduation.tar", False),
    # (r"E:/Code/RiceLodging/datasets/Meiju1_2_Lingtangkou/abnormal-03.20-7-640-0.1-0.6-0.2-0.2-v6/labels", "Labels.tar.gz", True),
    # (r"C:\path\to\your\data\TwentyGB_Data", "TwentyGB_Data/", True), # 假设这是20GB数据的文件夹
]

# --- 配置结束 ---
def convert_to_cygwin_path(windows_path):
    """将 Windows 路径转换为 Cygwin 路径"""
    if not windows_path:
        return ""
    path = windows_path.replace("\\", "/")
    if path[1] == ":":
        path = f"/cygdrive/{path[0].lower()}{path[2:]}"
    return path

def run_rsync(local_path, remote_target, is_directory):
    """执行单个 rsync 命令"""
    # 检查本地路径是否存在
    if not os.path.exists(local_path):
        print(f"错误：本地路径未找到: {local_path}")
        return False

    # 对 Windows 路径进行处理，确保 rsync 能正确识别
    # (rsync on Windows often expects cygwin-style paths, but modern versions might handle Windows paths better)
    # For simplicity, let's assume rsync handles the provided path. If errors occur, path conversion might be needed.

    # 构建远程目标路径
    # os.path.join 在这里可能不适用于构建 user@host:path 格式，我们直接拼接
    remote_dest = f"{USERNAME}@{SERVER_IP}:{REMOTE_BASE_PATH}{remote_target}"

    cygwin_local_path = convert_to_cygwin_path(local_path)

    # 构建 rsync 命令列表
    command = [
        RSYNC_PATH,
        "-avz",        # 归档, 详细, 压缩
        "--progress",  # 显示进度
        "-e",
        ssh_command,
        # "--info=progress2", # 或者使用这个获取整体进度 (如果rsync版本支持)
        cygwin_local_path,    # 本地源
        remote_dest    # 远程目标
    ]

    print(f"\n--- 开始同步: {local_path} -> {remote_dest} ---")
    # 使用 shlex.quote 确保路径中的空格等被正确处理 (主要用于打印命令)
    # print(f"执行命令: {' '.join(shlex.quote(c) for c in command)}") 

    try:
        # 使用 shell=False 更安全，直接传递列表给 Popen/run
        # 使用 subprocess.run 等待命令完成
        # text=True (或者 encoding='utf-8') 使输出为文本而非字节
        # check=True 会在 rsync 返回非零退出码时抛出异常
        result = subprocess.run(command, check=True, text=True, encoding='utf-8', errors='replace') 
        print(f"--- 成功同步: {local_path} ---")
        return True
    except FileNotFoundError:
         print(f"错误: '{RSYNC_PATH}' 命令未找到。请确认 rsync 已安装并在系统 PATH 中，或脚本中指定了正确路径。")
         return False
    except subprocess.CalledProcessError as e:
        # rsync 失败 (例如，网络问题，权限问题等)
        print(f"错误: rsync 同步失败 {local_path} ，退出码: {e.returncode}")
        # e.output, e.stdout, e.stderr 可能包含更多信息，但默认不捕获
        print("请检查网络连接、服务器路径、权限设置。")
        return False
    except Exception as e:
         print(f"同步过程中发生意外错误 {local_path}: {e}")
         return False

# --- 主执行逻辑 ---
all_successful = True
for local, remote, is_dir in UPLOAD_ITEMS:
    # 注意：Python 的路径字符串中，反斜杠 `\` 是转义字符。
    # 使用原始字符串 `r"C:\path..."` 或者双反斜杠 `"C:\\path..."`
    local_path_normalized = os.path.normpath(local) # 标准化路径表示

    if not run_rsync(local_path_normalized, remote, is_dir):
        all_successful = False
        print(f"!!! 上传失败: {local_path_normalized}。你可以尝试重新运行脚本来续传。!!!")
        # 选择：遇到错误时停止 (取消下一行注释) 或继续尝试其他文件
        # sys.exit(1) # 强制退出脚本

if all_successful:
    print("\n--- 所有配置的上传任务已成功完成！ ---")
else:
    print("\n--- 部分上传任务失败。请检查上面的日志。 ---")
    print("提示：由于 rsync 支持断点续传，您可以直接重新运行此脚本来尝试完成未完成或失败的传输。")
