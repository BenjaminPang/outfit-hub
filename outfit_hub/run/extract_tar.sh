#!/bin/bash
DATA_DIR="./data"

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Cannot find directory $DATA_DIR"
    exit 1
fi


for d in "$DATA_DIR"/*/; do 
    if [ -d "$d" ]; then 
        echo "📂 Processing dataset: $d..."
        
        # 进入数据集子目录
        cd "$d"
        
        # 创建 images 文件夹
        mkdir -p images
        
        # 查找并解压所有 .tar 文件
        # 使用 -C 指定解压到刚刚创建的 images 文件夹
        ls *.tar >/dev/null 2>&1 && for f in *.tar; do 
            echo "  📦 Extracting $f..."
            tar -xf "$f" -C images/
        done
        
        # 返回最初运行脚本时的根目录
        cd - > /dev/null
    fi
done