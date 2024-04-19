for d in data/*/ ; do
    if [ ${#d} -gt 9 ]; then
        continue
    fi
    rm -rf "$d/ifo001"
    rm -rf "$d/ifo002"
    rm -rf "$d/ifo003"
    rm -rf "$d/timeshift.yaml"
done