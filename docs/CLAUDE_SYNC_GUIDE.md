# Hướng dẫn Đồng bộ Lịch sử Claude Code (Git-based)

Tài liệu này hướng dẫn cách sử dụng Git để đồng bộ lịch sử chat của Claude Code giữa máy Server (Vast.ai) và máy cá nhân của bạn một cách an toàn, tránh bị mất dữ liệu như khi dùng các công cụ đồng bộ file (Resilio Sync, Dropbox).

## 1. Tại sao dùng Git?
- **Không bao giờ bị đè mất dữ liệu**: Git sẽ tự động gộp (merge) lịch sử từ 2 máy thay vì xóa file cũ.
- **Có lịch sử phiên bản**: Bạn có thể xem lại hoặc khôi phục bất kỳ phiên bản nào.
- **Bảo mật**: Chỉ đồng bộ nội dung chat, không đồng bộ thông tin đăng nhập máy (vì mỗi máy có token riêng).

---

## 2. Thiết lập trên Server (Đã hoàn tất)

Tôi đã thiết lập sẵn các thành phần sau trên Server:
- **Git Repo**: Sử dụng Repo có sẵn tại `/workspace/qwen_calligraphy_lora/claude_config`.
- **Symlink**: Đã liên kết `~/.claude` tới thư mục workspace nói trên (để lịch sử đi kèm ổ đĩa workspace).
- **.gitignore**: Đã cấu hình trong repo workspace.
- **Script Sync**: Chạy lệnh `~/claude-sync.sh` để bắt đầu quá trình Lưu -> Tải -> Gộp -> Đẩy dữ liệu.
- **Remote Origin**: Đã cấu hình tới repository của bạn (HTTPS).
- **Credential Helper**: Đã bật `store` để ghi nhớ token GitHub.

---

## 3. Thiết lập kết nối (Làm 1 lần duy nhất)

Bạn cần tạo một nơi chứa dữ liệu trên mây:
1. Tạo một repository **PRIVATE** (Riêng tư) trên [GitHub](https://github.com/new). (Đã tạo: `git@github.com:DoTuanPhong/.claude_qwen_lora.git`)
2. Chạy lệnh sau trên Server để kết nối:
   ```bash
   cd ~/.claude
   git remote add origin git@github.com:DoTuanPhong/.claude_qwen_lora.git
   ```

---

## 4. Thiết lập trên Máy cá nhân

### A. Nếu là máy Linux/macOS
Trên các hệ điều hành Unix, bạn thực hiện các bước sau:
1. **Khởi tạo và kết nối**:
   ```bash
   cd ~/.claude
   git init
   git remote add origin <URL-repo-private-vừa-tạo>
   git config pull.rebase true
   ```
2. **Cấu hình `.gitignore`**: Tạo file `~/.claude/.gitignore` với nội dung tương tự trên server.
3. **Cài đặt Script Sync**: Cài đặt script `claude-sync.sh` để bắt đầu quá trình đồng bộ.

### B. Nếu là máy Windows
Trên Windows, Claude Code thường được lưu tại: `C:\Users\Admin\.claude` (hoặc `%USERPROFILE%\.claude`).

1. **Khuyên dùng Git Bash**:
   - Cài đặt [Git for Windows](https://git-scm.com/download/win).
   - Mở **Git Bash** và thực hiện các bước y hệt như máy Linux ở trên (trong Git Bash, đường dẫn vẫn dùng dấu `/`).
2. **Cấu hình Line Endings**: Chạy lệnh này 1 lần trên Windows để tránh lỗi định dạng file JSONL:
   ```bash
   git config --global core.autocrlf input
   ```
3. **Script cho PowerShell (Nếu không muốn dùng Git Bash)**: Tạo file `claude-sync.ps1` trong folder người dùng:
   ```powershell
   cd "$env:USERPROFILE\.claude"
   git add .
   git commit -m "Sync history $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
   git pull --rebase origin master
   git push origin master
   ```

---

## 5. Thiết lập Instance mới (Khi đổi máy Server Vast.ai)

Tùy thuộc vào việc instance mới có **dùng chung ổ đĩa workspace** với instance cũ hay không, bạn chọn 1 trong 2 cách sau:

### Cách A: Nếu DÙNG CHUNG Workspace (ổ đĩa `/workspace` được giữ nguyên)
Trường hợp này bạn không cần clone lại dữ liệu, chỉ cần tạo liên kết (symlink) để Claude CLI tìm thấy folder:
```bash
# Xóa folder mặc định nếu có
rm -rf ~/.claude
# Tạo liên kết tới folder trong workspace
ln -s /workspace/qwen_calligraphy_lora/claude_config ~/.claude
```

### Cách B: Nếu KHÔNG dùng chung Workspace (Instance hoàn toàn mới)
Bạn cần "kéo" lịch sử từ GitHub về máy mới:
1. Cài đặt và thực hiện đăng nhập Claude (`claude` lệnh để login).
2. **Cấu hình Git GLOBAL trước** (tránh lỗi "Make sure you configure your user.name and user.email in git" khi commit/sync):
   ```bash
   git config --global user.name "DoTuanPhong"
   git config --global user.email "phong02468@gmail.com"
   git config --global init.defaultBranch main
   git config --global core.ignorecase false
   ```
3. **Liên kết thư mục cấu hình vào Workspace** (để hiển thị trên Source Control của IDE và tránh mất dữ liệu khi phân vùng `/root` bị xóa):
   ```bash
   # Tạo thư mục cấu hình trong workspace
   mkdir -p /workspace/qwen_calligraphy_lora/claude_config

   # Xóa thư mục mặc định nếu có
   rm -rf ~/.claude

   # Tạo liên kết symlink
   ln -s /workspace/qwen_calligraphy_lora/claude_config ~/.claude
   ```
4. **Khởi tạo Git và kéo dữ liệu về**:
   ```bash
   cd ~/.claude
   git init
   git remote add origin https://github.com/DoTuanPhong/.claude_qwen_lora.git
   git fetch origin
   git reset --hard origin/master
   ```
5. **Cài đặt lại script sync**:
   - Copy nội dung script `claude-sync.sh` (mục 6) vào `~/claude-sync.sh`.
   - Chạy `chmod +x ~/claude-sync.sh`.

**Vì sao phải set Global?** Nếu chỉ set user.name/user.email ở local (trong `.git/config` của repo), khi chạy `git` ở directory khác (ví dụ `/tmp`, sub-folder, repo khác) thì Git sẽ không tìm thấy user info → báo lỗi. Global config ở `/root/.gitconfig` là fallback cho mọi git operation trên máy, bất kể working directory. Xem thêm `docs/troubleshoot_git.md`.


---

## 6. Script Đồng bộ (`claude-sync.sh`)

Sử dụng script này hàng ngày để đảm bảo dữ liệu luôn được cập nhật:

```bash
#!/bin/bash
CLAUDE_DIR="$HOME/.claude"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

cd "$CLAUDE_DIR" || exit

# 1. Lưu lịch sử máy hiện tại
git add .
if git diff-index --quiet HEAD --; then
    echo "Không có thay đổi mới."
else
    git commit -m "Sync history [$TIMESTAMP]"
fi

# 2. Đồng bộ với Cloud
if git remote | grep -q 'origin'; then
    git pull --rebase origin master
    git push origin master
    echo "=== Đồng bộ thành công! ==="
else
    echo "Lỗi: Chưa cấu hình 'origin'. Hãy chạy 'git remote add origin <URL>'."
fi
```

## 7. Lưu ý quan trọng
- Luôn chạy script này **trước khi bắt đầu làm việc** trên một máy mới để đảm bảo bạn có lịch sử mới nhất.
- Luôn chạy script này **sau khi kết thúc làm việc** để đẩy dữ liệu lên mây.
