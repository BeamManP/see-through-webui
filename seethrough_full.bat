@echo off
call conda activate see_through
set HF_HOME=F:\seethrough\.hf_cache
cd /d F:\seethrough
python tools\seethrough_easy.py --full %*
pause
