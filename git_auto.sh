#!/bin/bash
# 걸파 툴 모음 (tools_girlsbandparty) — add -> commit -> pull -> push 자동화
# 사용법: ./git_auto.sh "커밋메시지"   (생략 시 자동 생성)
# 참고: rank/data.json 은 GitHub Actions(bot)가 매일 자동 갱신하므로,
#       push 전 pull 로 그 커밋을 먼저 받아온다(충돌 방지).

set +H  # '!' 히스토리 확장 비활성화

# 0. 스크립트 자신의 폴더로 이동 (어디서 실행하든 올바른 저장소에서 동작)
cd "$(dirname "$0")" || exit 1

# 1. 변경사항 추가
echo "[+] Adding changes..."
git add .

# 2. 커밋 (변경 없으면 건너뜀 — 빈 커밋 에러 방지)
COMMIT_MSG=${1:-"Auto commit - $(date '+%Y-%m-%d %H:%M:%S')"}
if git diff --cached --quiet; then
    echo "[i] 커밋할 변경사항이 없습니다. 커밋을 건너뜁니다."
else
    echo "[+] Committing with message: $COMMIT_MSG"
    git commit -m "$COMMIT_MSG"
fi

# 3. Pull (원격 = Actions가 올린 data.json 등 병합)
echo "[+] Pulling from remote..."
git pull --no-rebase --no-edit
if [ $? -ne 0 ]; then
    echo ""
    echo "###################################################"
    echo " [!!!] 에러: 머지 충돌(Conflict) 감지. 직접 해결 후 다시 실행하세요."
    echo "###################################################"
    read -p "[?] 창을 닫으려면 Enter를 누르세요..." _
    exit 1
fi

# 4. Push
echo "[+] Pushing to remote..."
git push
if [ $? -ne 0 ]; then
    echo ""
    echo " [!!!] 에러: Push 실패!"
    echo "     - .github/workflows/*.yml 을 수정했다면 git 토큰에 'workflow' 권한이 필요합니다."
    echo "       (그 경우 워크플로는 GitHub 웹에서 수정하세요.)"
    echo "     - 그 외엔 권한/원격 설정을 확인하세요."
    read -p "[?] 창을 닫으려면 Enter를 누르세요..." _
    exit 1
fi

echo ""
echo "[OK] 커밋 + 푸시 완료!"
exit 0
