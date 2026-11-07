@echo off
echo Pushing Crop Yield API to GitHub (https://github.com/Premaben07/crop--yeild--api)...
git push -u origin main
if %errorlevel% equ 0 (
    echo.
    echo ==============================================
    echo Successfully pushed to GitHub!
    echo Visit: https://github.com/Premaben07/crop--yeild--api
    echo ==============================================
) else (
    echo.
    echo ==============================================
    echo Push encountered an issue. Check your credentials.
    echo ==============================================
)
pause
