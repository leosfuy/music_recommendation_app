
#test MSD40000.sql
import os
file_path = "/Users/leeangel/Desktop/資料庫專題/MSD40000.sql"

print(f"🕵️‍♀️ 正在偵查檔案：{os.path.basename(file_path)} ...")
print("-" * 50)

try:
    # 使用 errors='replace' 避免遇到奇怪編碼報錯
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        # 只讀取前 50 行
        for i in range(50):
            line = f.readline()
            if not line: break # 檔案結束就停止
            
            # 印出每一行，並去除前後空白
            print(f"[Line {i+1}] {line.strip()[:200]}") # 每行只印前200字，避免洗版
            
            # 如果這行太長（代表是資料行），我們把它截斷顯示
            if len(line) > 200:
                print("... (後面太長省略) ...")

    print("-" * 50)
    print("✅ 偵查完畢！")

except FileNotFoundError:
    print("❌ 找不到檔案，請確認路徑對不對！")
except Exception as e:
    print(f"❌ 發生錯誤：{e}")