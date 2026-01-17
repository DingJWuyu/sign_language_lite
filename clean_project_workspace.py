import os
import shutil

def clean_project_workspace():
    """
    清理当前目录下的缓存、运行日志和模型检查点。
    """
    # 定义需要删除的目标：(路径, 是否仅删除内容)
    # True 表示保留文件夹本身只删内容，False 表示连文件夹一起删
    targets = {
        "__pycache__": False,
        "runs": False,
        "checkpoints": True
    }

    current_dir = os.getcwd()
    print(f"开始清理目录: {current_dir}\n" + "-"*30)

    for name, content_only in targets.items():
        path = os.path.join(current_dir, name)
        
        if os.path.exists(path):
            try:
                if content_only:
                    # 仅删除文件夹内的所有内容，保留文件夹
                    for item in os.listdir(path):
                        item_path = os.path.join(path, item)
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    print(f"[已清理内容] {name}/")
                else:
                    # 删除整个文件夹
                    shutil.rmtree(path)
                    print(f"[已完全删除] {name}/")
            except Exception as e:
                print(f"[错误] 无法处理 {name}: {e}")
        else:
            print(f"[跳过] {name} 不存在")

    print("-"*30 + "\n清理完成！")

if __name__ == "__main__":
    clean_project_workspace()