#!/bin/bash

# 1. 直接指定你要处理的文件夹路径
TARGET_DIR="data/polyvore_outfits_disjoint"

# 检查目录是否存在
if [ -d "$TARGET_DIR" ]; then
    echo "📂 Processing dataset: $TARGET_DIR..."
    
    # 使用子词法环境 (括号)，这样执行完后会自动回到当前目录，不需要 cd -
    (
        cd "$TARGET_DIR" || exit
        
        # 创建 images 文件夹
        mkdir -p images
        
        # 查找是否有 .tar 文件
        # 使用 compgen 检查比 ls 更优雅，防止没有文件时报错
        if ls *.tar >/dev/null 2>&1; then
            for f in *.tar; do 
                echo "  📦 Extracting $f into images/ ..."
                tar -xf "$f" -C images/
            done
            echo "✅ Extraction complete."
        else
            echo "⚠️  No .tar files found in $TARGET_DIR"
        fi
    )
else
    echo "❌ Error: Directory $TARGET_DIR does not exist."
fi