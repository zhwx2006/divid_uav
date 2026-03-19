@echo off
chcp 65001 > nul
echo ==============================
echo 无人机训练脚本（Win11终极版）
echo ==============================

:: 设置环境变量关闭冗余提示
set SUPPRESS_MA_PROMPT=1

:: 激活conda环境（确保uav_epc环境存在）
call conda activate uav_epc

:: 切换到训练目录（你的实际路径）
cd /d C:\Users\22895\Desktop\uav_project\UAV-path-planning

:: 启动训练脚本
python train_uav_food_collection.py

:: 训练完成提示
echo.
echo 训练结束！按任意键退出...
pause > nul