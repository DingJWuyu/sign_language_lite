@echo off
chcp 65001 >nul
echo ============================================
echo    轻量化手语识别系统 - 快速启动
echo ============================================
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python或Anaconda
    pause
    exit /b 1
)

:: 显示菜单
:menu
echo.
echo 请选择操作:
echo   1. 安装依赖
echo   2. 创建演示数据
echo   3. 开始训练
echo   4. 运行推理
echo   5. 实时演示 (需要摄像头)
echo   6. 检查环境
echo   0. 退出
echo.
set /p choice="请输入数字 (0-6): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto demo_data
if "%choice%"=="3" goto train
if "%choice%"=="4" goto inference
if "%choice%"=="5" goto realtime
if "%choice%"=="6" goto check
if "%choice%"=="0" goto end

echo 无效的选择，请重试
goto menu

:install
echo.
echo [1/2] 安装PyTorch (CUDA 12.1版本)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.
echo [2/2] 安装其他依赖...
pip install -r requirements_lite.txt
echo.
echo 安装完成!
pause
goto menu

:demo_data
echo.
echo 创建演示数据...
python data_sampling.py --mode demo
echo.
echo 演示数据创建完成!
pause
goto menu

:train
echo.
echo 开始训练模型...
python train.py
echo.
pause
goto menu

:inference
echo.
echo 运行推理测试...
if exist "checkpoints\best_model.pth" (
    python inference.py --model_path checkpoints\best_model.pth
) else (
    echo [警告] 未找到训练好的模型，请先训练
)
echo.
pause
goto menu

:realtime
echo.
echo 启动实时演示...
echo 注意: 需要摄像头和训练好的模型
python realtime_demo.py
echo.
pause
goto menu

:check
echo.
echo ========== 环境检查 ==========
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}');"
echo.
echo ========== 依赖检查 ==========
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import einops; print('einops: OK')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import cv2; print(f'opencv: {cv2.__version__}')"
echo.
echo ========== 数据检查 ==========
if exist "data\CSL_Daily_lite\labels.train" (
    echo 训练数据: 已就绪
) else (
    echo 训练数据: 未找到 (请运行"创建演示数据")
)
if exist "pretrained_weight\mt5-small\config.json" (
    echo mT5模型: 已就绪
) else (
    echo mT5模型: 未找到 (请下载mt5-small模型)
)
echo.
pause
goto menu

:end
echo.
echo 再见!
exit /b 0
