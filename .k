export RELEASE_LOG_LEVEL=DEBUG

GITHUB_TOKEN=$(cat "$HOME/.github_access_token")
export GITHUB_TOKEN

refresh() {
    git fetch --tag
    rm ./*.pickle
}

go() {
    #release -p "$PWD/release.py#The script" "README.md" && jq -r .release_message .release.json
    release -r -n "$PWD/release.py#The script" "README.md"
}
