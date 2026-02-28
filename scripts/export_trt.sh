#!/usr/bin/env bash
#
# Export SAM 3 ONNX models to TensorRT engines.
#
# Usage:
#   # Convert everything
#   ./export_trt.sh
#
#   # Convert specific network(s)
#   ./export_trt.sh backboneImageEncoder geometryEncoder
#
#   # List available networks
#   ./export_trt.sh --list
#
#   # Custom ONNX input / engine output directories
#   ONNX_DIR=./onnx_out ENGINE_DIR=./engines ./export_trt.sh backboneTextEncoder

set -euo pipefail

ONNX_DIR="${ONNX_DIR:-.}"
ENGINE_DIR="${ENGINE_DIR:-.}"

AVAILABLE_NETWORKS=(
    backboneImageEncoder
    backboneTextEncoder
    geometryEncoder
    transformerDetector
)

# ── Network-specific trtexec configs ──────────────────────────────────────────

run_backboneImageEncoder() {
    trtexec \
        --onnx="${ONNX_DIR}/backboneImageEncoder.onnx" \
        --saveEngine="${ENGINE_DIR}/backboneImageEncoder.engine" \
        --minShapes=image:1x3x1008x1008 \
        --optShapes=image:1x3x1008x1008 \
        --maxShapes=image:1x3x1008x1008
}

run_backboneTextEncoder() {
    trtexec \
        --onnx="${ONNX_DIR}/backboneTextEncoder.onnx" \
        --saveEngine="${ENGINE_DIR}/backboneTextEncoder.engine" \
        --minShapes=text_tokens:1x32 \
        --optShapes=text_tokens:1x32 \
        --maxShapes=text_tokens:1x32
}

run_geometryEncoder() {
    trtexec \
        --onnx="${ONNX_DIR}/geometryEncoder.onnx" \
        --saveEngine="${ENGINE_DIR}/geometryEncoder.engine" \
        --minShapes=points:1x1x2,points_mask:1x1,points_label:1x1,boxes:32x1x4,boxes_mask:1x32,boxes_labels:32x1,seq_first_img_feats:72x72x1x256,seq_first_img_pos_embeds:72x72x1x256 \
        --optShapes=points:32x1x2,points_mask:1x32,points_label:32x1,boxes:32x1x4,boxes_mask:1x32,boxes_labels:32x1,seq_first_img_feats:72x72x1x256,seq_first_img_pos_embeds:72x72x1x256 \
        --maxShapes=points:32x1x2,points_mask:1x32,points_label:32x1,boxes:32x1x4,boxes_mask:1x32,boxes_labels:32x1,seq_first_img_feats:72x72x1x256,seq_first_img_pos_embeds:72x72x1x256
}

run_transformerDetector() {
    trtexec \
        --onnx="${ONNX_DIR}/transformer.onnx" \
        --saveEngine="${ENGINE_DIR}/transformer.engine" \
        --minShapes=backbone_fpn_0:1x256x288x288,backbone_fpn_1:1x256x144x144,backbone_fpn_2:1x256x72x72,vision_pos_enc_2:1x256x72x72,prompt:97x1x256,prompt_mask:1x97 \
        --optShapes=backbone_fpn_0:1x256x288x288,backbone_fpn_1:1x256x144x144,backbone_fpn_2:1x256x72x72,vision_pos_enc_2:1x256x72x72,prompt:97x1x256,prompt_mask:1x97 \
        --maxShapes=backbone_fpn_0:1x256x288x288,backbone_fpn_1:1x256x144x144,backbone_fpn_2:1x256x72x72,vision_pos_enc_2:1x256x72x72,prompt:97x1x256,prompt_mask:1x97
}

# ── Helpers ───────────────────────────────────────────────────────────────────

list_networks() {
    echo "Available networks:"
    for net in "${AVAILABLE_NETWORKS[@]}"; do
        echo "  - ${net}"
    done
}

is_valid_network() {
    local name="$1"
    for net in "${AVAILABLE_NETWORKS[@]}"; do
        [[ "$net" == "$name" ]] && return 0
    done
    return 1
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [NETWORK ...]

Convert SAM 3 ONNX models to TensorRT engines.
If no networks are specified, all networks are converted.

Options:
  --list        List available networks and exit
  -h, --help    Show this help and exit

Environment variables:
  ONNX_DIR      Directory containing .onnx files  (default: .)
  ENGINE_DIR    Directory for output .engine files (default: .)

Available networks: $(IFS=', '; echo "${AVAILABLE_NETWORKS[*]}")
EOF
}

# ── Main ──────────────────────────────────────────────────────────────────────

networks_to_convert=()

for arg in "$@"; do
    case "$arg" in
        --list)
            list_networks
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if is_valid_network "$arg"; then
                networks_to_convert+=("$arg")
            else
                echo "Error: unknown network '${arg}'" >&2
                echo "Available: $(IFS=', '; echo "${AVAILABLE_NETWORKS[*]}")" >&2
                exit 1
            fi
            ;;
    esac
done

# Default: convert all
if [[ ${#networks_to_convert[@]} -eq 0 ]]; then
    networks_to_convert=("${AVAILABLE_NETWORKS[@]}")
fi

mkdir -p "${ENGINE_DIR}"

echo "=== SAM 3 ONNX → TensorRT ==="
echo "ONNX dir:   ${ONNX_DIR}"
echo "Engine dir:  ${ENGINE_DIR}"
echo "Networks:    ${networks_to_convert[*]}"
echo ""

failed=()
for net in "${networks_to_convert[@]}"; do
    echo "──── Converting: ${net} ────"
    if "run_${net}"; then
        echo "✓ ${net} done"
    else
        echo "✗ ${net} FAILED" >&2
        failed+=("$net")
    fi
    echo ""
done

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "=== FAILED: ${failed[*]} ===" >&2
    exit 1
fi

echo "=== All done ==="
