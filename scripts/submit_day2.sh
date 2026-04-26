#!/usr/bin/env bash
# Submit day-2 candidates in priority order with adaptive pauses.
# Stops at 20 submissions (Kaggle daily limit).
set -u

COMP="datathon-2026-round-1"
MAX_SUBS=20
PAUSE_SEC=45

cd "$(dirname "$0")/.."

submit() {
    local file="$1"
    local msg="$2"
    if ! [[ -f "$file" ]]; then
        echo "MISSING: $file" >&2
        return 1
    fi
    echo "[$(date -u +%H:%M:%S)] Submitting $file -- $msg"
    kaggle competitions submit -c "$COMP" -f "$file" -m "$msg" 2>&1 | tail -3
    return $?
}

# Priority-ordered plan. Highest gain first.
DECLS=(
    "outputs/candidates/day2/sub_quad_min.csv|quad min rev(1.28,1.40) cog(1.39,1.445)"
    "outputs/candidates/day2/sub_predicted_1p323_1p384_1p365_1p484.csv|predicted 1D opt rev(1.323,1.384) cog(1.365,1.484)"
    "outputs/candidates/day2/sub_refined_near_best.csv|refined near best rev(1.325,1.385) cog(1.385,1.495)"
    "outputs/candidates/day2/sub_quad_alt3.csv|quad alt3 rev(1.30,1.39) cog(1.39,1.455)"
    "outputs/candidates/day2/sub_quad_alt.csv|quad alt rev(1.29,1.395) cog(1.385,1.45)"
    "outputs/candidates/day2/sub_quad_alt2.csv|quad alt2 rev(1.31,1.385) cog(1.395,1.47)"
    "outputs/candidates/day2/sub_cog23_down.csv|cog23 down rev(1.33,1.39) cog(1.37,1.50)"
    "outputs/candidates/day2/sub_cog24_up.csv|cog24 up rev(1.33,1.39) cog(1.38,1.51)"
    "outputs/candidates/day2/sub_cog23_up.csv|cog23 up rev(1.33,1.39) cog(1.39,1.50)"
    "outputs/candidates/day2/sub_refined_v2.csv|refined v2 rev(1.32,1.385) cog(1.38,1.495)"
    "outputs/candidates/day2/sub_custom_doy_wtd_hl2_all_dow00.csv|custom hl2 all dow00"
    "outputs/candidates/day2/sub_custom_doy_wtd_hl2_all_dow50.csv|custom hl2 all dow50"
    "outputs/candidates/day2/sub_blend_sample_vs_custom_w70.csv|blend sample0.7 custom0.3"
    "outputs/candidates/day2/sub_blend_sample_vs_custom_w50.csv|blend sample0.5 custom0.5"
    "outputs/candidates/day2/sub_fine1_1p325_1p375_1p370_1p490.csv|fine1 1.325/1.375 1.37/1.49"
    "outputs/candidates/day2/sub_fine3_1p325_1p380_1p375_1p485.csv|fine3 1.325/1.38 1.375/1.485"
    "outputs/candidates/day2/sub_fine5_1p320_1p385_1p375_1p490.csv|fine5 1.32/1.385 1.375/1.49"
    "outputs/candidates/day2/sub_higher_cog24_1p330_1p380_1p380_1p520.csv|higher cog24 1.52"
    "outputs/candidates/day2/sub_custom_doy_avg_1922_wtd2022_heavy_dow00.csv|custom 2022heavy dow00"
    "outputs/candidates/day2/sub_blend_sample_vs_custom_w30.csv|blend sample0.3 custom0.7"
)

n=0
for entry in "${DECLS[@]}"; do
    IFS='|' read -r file msg <<<"$entry"
    submit "$file" "$msg"
    rc=$?
    n=$((n+1))
    if [[ $rc -ne 0 ]]; then
        echo "-> FAILED (rc=$rc). Pausing ${PAUSE_SEC}s before retry."
        sleep "$PAUSE_SEC"
        continue
    fi
    sleep "$PAUSE_SEC"
    if [[ $n -ge $MAX_SUBS ]]; then
        break
    fi
done

echo
echo "All submissions attempted. Check status:"
kaggle competitions submissions -c "$COMP" | head -25
