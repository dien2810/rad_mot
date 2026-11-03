# RAD-MOT: Risk-Aware Dispatch cho SAM trong theo vết đa đối tượng

Mã nguồn cho đề tài *Learning When to Call SAM: Risk-Aware Selective Mask Propagation for Multi-Object Tracking in Dense Motorbike Traffic*.

Ý tưởng: tracker nhẹ (ByteTrack) chạy ở mọi khung. SAM 3 chỉ được gọi để sửa ID ở những khoảnh khắc rủi ro. Selective Mask Propagation (SMP) quyết định "khi nào gọi" bằng một ngưỡng margin cố định. Đề tài thay ngưỡng đó bằng một bộ dự đoán rủi ro học được, nhãn tự sinh từ ground truth.

## Cấu trúc

```
rad/
  io/mot.py              đọc/ghi định dạng MOT, FastTracker (9 cột), tìm sequence
  tracker/bytetrack.py   ByteTrack cài lại + log margin, gap, mật độ cho từng lần gán
  eval/idsw.py           CLEAR-MOT + IDF1, trả về từng sự kiện đổi ID (đã đối chiếu khớp với motmetrics)
  eval/trackeval_runner.py  HOTA/MOTA/IDF1 qua TrackEval
  sam/smp.py             Selective Mask Propagation cài lại (cửa sổ, mồi, IoMA, SWAP)
  sam/backends.py        OracleBackend (CPU, theo GT) và Sam3Backend (GPU)
  dispatch/candidates.py sự kiện ứng viên + đặc trưng (gồm ứng viên "birth" cho lỗi đứt track)
  dispatch/labels.py     nhãn y_risk (từ GT) và y_fix (chạy thử cửa sổ)
  dispatch/model.py      gradient boosting, out-of-fold theo sequence
  dispatch/policy.py     các chính sách gọi SAM: smp, smp+birth, density, learned, random
  pilot/analysis.py      phân tích H1/H2 theo mật độ, biểu đồ, báo cáo
  synth/simulate.py      dữ liệu giao thông giả lập (chỉ để kiểm tra mã)
  pipeline.py            ghép các bước
scripts/                 các lệnh chạy (xem bên dưới), gồm estimate_gpu_time.py để ước lượng giờ GPU
configs/default.yaml     toàn bộ tham số
notebooks/colab_pipeline.ipynb  chạy trên Colab/Kaggle
tests/                   kiểm thử nhanh (python -m pytest -q)
```

## Cài đặt

Phần CPU (tracker, pilot, mô hình dispatch):

```bash
pip install -r requirements.txt
```

Phần GPU (SAM 3, YOLO). SAM 3 cần Python ≥ 3.12, PyTorch ≥ 2.7, CUDA ≥ 12.6, và quyền tải checkpoint:
1. Xin quyền tại https://huggingface.co/facebook/sam3, rồi chạy `hf auth login`.
2. Cài đặt:
   ```bash
   git clone https://github.com/facebookresearch/sam3 && pip install -e sam3
   pip install ultralytics
   ```

GPU có bfloat16 phần cứng (L4, A100, RTX 30xx/40xx) chạy SAM 3 nhanh nhất. T4 chạy được nhưng chậm hơn, xem mục "Chạy với Colab miễn phí" bên dưới.

> Lưu ý ổ đĩa: ổ chứa thư mục này trên máy bạn chỉ còn khoảng 4 GB trống. Hãy tải dữ liệu và chạy phần GPU trên Colab/Kaggle (lưu vào Google Drive), đừng tải về đây.

## Chạy với Colab miễn phí (GPU T4)

Không bắt buộc Colab Pro. Phần lớn công việc chạy bằng CPU, chỉ YOLO và SAM 3 cần GPU.

| Bước | Cần gì | T4 miễn phí |
|---|---|---|
| Thống kê dữ liệu, pilot H1/H2, mô hình dispatch (VOS oracle) | CPU | không tốn giờ GPU |
| YOLO detection | GPU | được; dùng `--half`, model `yolo11l` |
| SAM 3 tracker (bản đề tài dùng) | GPU | theo mã nguồn thì không có gì chặn (chưa chạy thử thực tế). bfloat16 bị giả lập trên T4 nên sẽ chậm hơn L4/A100; đo trước bằng `smoke_sam3.py` |
| SAM 3.1 Object Multiplex | GPU Ampere trở lên | khả năng cao **không** chạy: mã ép dùng kernel FlashAttention, vốn cần GPU Ampere trở lên |

Cách làm với T4:
1. Đo tốc độ: `python scripts/smoke_sam3.py ... --precision auto`, rồi chạy lại với `--precision fp16` (thử nghiệm). Chọn chế độ nhanh hơn nếu độ bám mask không giảm.
2. Ước lượng giờ GPU trước khi chạy:
   ```bash
   python scripts/estimate_gpu_time.py --ms-per-frame <số đo được>
   ```
   Script lấy số khung SAM phải xử lý từ các lần chạy oracle trên CPU.
3. Phát triển và quét chính sách bằng VOS oracle (CPU). Chỉ dùng SAM 3 để xác nhận vài cấu hình cuối, ví dụ `smp@0.05` và `learned@1.0`, trên vài sequence.
4. Chạy SAM 3 theo từng sequence (`--seqs`) và ghi kết quả vào Drive sau mỗi lần. Phiên miễn phí tối đa 12 giờ, có thể bị ngắt khi không hoạt động, và không được bảo đảm có GPU.
5. Dữ liệu khoảng 14,8 GB: tải vào `/content` (ổ tạm của phiên), không tải vào Drive miễn phí (15 GB).

## Quy trình theo tuần

### Tuần 1–2: dữ liệu

Tải FastTracker benchmark (dataset `Hamidreza-Hashemp/FastTracker-Benchmark` trên Hugging Face). Bộ dữ liệu gồm 12 tệp zip, mỗi sequence một tệp (0,5–1,6 GB), bên trong có `img1/`, `gt/gt.txt`, `video/`, `seqinfo.ini`.

```bash
# chỉ GT + seqinfo (vài MB): đọc thẳng trong zip qua HTTP Range, không tải cả zip
python scripts/download_fasttracker.py --out datasets/FastTracker --mode gt
# ảnh cho giai đoạn GPU: tải và giải nén từng zip rồi xoá zip (bỏ video/)
python scripts/download_fasttracker.py --out datasets/FastTracker --mode full --seqs task_tunnel task_night_far_objects
```

Sau đó đếm lớp và mật độ:

```bash
python scripts/dataset_stats.py --data datasets/FastTracker/train --out outputs/stats
```

Việc đầu tiên là xem tỷ lệ `motorbike` (và `pm`). Mã lớp trong GT của FastTracker chạy từ 1 (bảng đầy đủ trong `rad/classes.py`, lấy từ bản TrackEval trong repo FastTracker): 1 person, 2 bus_small, 3 bus_big, 4 truck_small, 5 truck_big, 6 car, 7 bike, **8 motorbike**, **9 ignore_region**, 13 heavy_equipment, 14 pm. Toàn bộ code dùng bảng mã này (cả YOLO và dữ liệu tổng hợp). Nếu xe máy chiếm ít, đổi tên đề tài sang "dense mixed traffic" hoặc bổ sung TrafficMOT / video Việt Nam.

Lớp 9 là vùng bỏ qua, không phải đối tượng (`gt_ignore_classes: [9]` trong config). Ở mọi bước (GT, detection oracle hoặc YOLO, đánh giá TrackEval), vùng này được loại, và mọi box nằm phần lớn trong vùng (giao / diện tích box > `det_ignore_ioa` = 0,5, cùng khung) cũng bị loại. `dataset_stats` in số box vùng bỏ qua (`ignore_region_boxes`) và số đối tượng GT bị loại vì nằm trong vùng (`objects_inside_ignore`).

### Tuần 3–5: pilot H1/H2, chỉ cần CPU và GT

```bash
python scripts/run_pilot.py --data datasets/FastTracker/train --dets oracle --backend oracle --out outputs/pilot
python scripts/eval_trackeval.py --data datasets/FastTracker/train --results outputs/pilot --name bytetrack.txt
python scripts/eval_trackeval.py --data datasets/FastTracker/train --results outputs/pilot --name smp_oracle.txt
```

- `--dets oracle`: lấy detection từ GT (vật bị che nhiều thì điểm thấp hoặc bị bỏ). Cách này tách riêng bài toán liên kết khỏi chất lượng detector.
- `--backend oracle`: thay SAM bằng một "VOS hoàn hảo" bám theo GT. Đây là cận trên cho chất lượng SAM, dùng để đo cơ chế SMP mà chưa cần GPU.

- `--density gt` (mặc định): mật độ = số đối tượng GT trong khung, giống nhau ở mọi lần chạy (oracle, YOLO, SAM 3). `--density tracker`: số box của tracker. H1 và H2 luôn dùng cùng một thước đo; khung không có đối tượng tính là mật độ 0.
- Độ nhạy witness: `--no-witness` (chỉ cửa sổ do trigger mở) hoặc `--witness-iou 0.3` (chỉ mở witness cho track chồng lấn IoU > 0.3). Nên ghi ra thư mục khác, ví dụ `--out outputs/pilot_nowit`.

Kết quả nằm trong `outputs/pilot/` (`pilot_report.md`, các CSV, 3 biểu đồ PNG). Định nghĩa dùng chung:
- **kind**: `trigger` (do quy tắc margin/gap mở) hoặc `witness` (mở cho track có box chồng lên track bị trigger).
- **ran**: tìm được seed, SAM/VOS thực sự chạy. Cửa sổ SEED_FAIL không tốn khung SAM nào nên được tách riêng.
- **wasted**: đã chạy nhưng không đổi ID nào: kết quả khác SWAP, hoặc SWAP trùng với một cửa sổ khác đã sửa cùng cặp A↔B.

| File | Cột chính |
|---|---|
| `h1_low_margin_share` | tỷ lệ cặp gán có margin < 0.01 / 0.025 / 0.05, `n_pairs` |
| `h1_windows` | `n_windows`, `witness_share`, `seed_fail_share`, `n_ran`, `wasted_share_ran`, `mean_frames_ran`, `vos_frames` |
| `h1_windows_by_kind` | như trên, tách theo (mật độ, kind) |
| `h1_windows_per_frame` | `frames`, `windows_per_frame` (= trigger + witness), `windows_per_100_objects`, `trigger_windows_per_100_objects`, `vos_frames_per_frame` |
| `outcomes` | số cửa sổ theo kết quả × kind |
| `h2_by_type`, `h2_by_density` | mỗi lỗi đổi ID thật thuộc đúng một nhóm: `covered_by_window` (SAM chạy qua frame lỗi, ±3), `seed_fail_near` (chỉ có cửa sổ SEED_FAIL), `no_window`; `trigger_*` (quy tắc trigger có bắn trong ±5 frame); `fixed` (lỗi biến mất sau khi sửa, cùng đối tượng GT, ±5 frame); `n` |
| `h2_margin_of_missed` | lỗi `no_window` theo loại: `n_stage3` (track mới được gán ở bước 3 của ByteTrack, margin mặc định 1.0 nên không có nghĩa), `n_margin_valid` (bước 1–2), phân vị margin, `no_competitor_share` |
| `per_sequence` | `objects_per_frame` (GT), `tracker_boxes_per_frame`, IDSW/IDF1 trước và sau, `switches_fixed`, `switches_introduced`, `triggers`, `windows` = `windows_trigger` + `windows_witness`, `windows_ran`, `swaps_applied`, `vos_frames`, số cửa sổ theo kết quả |

Từng lỗi đổi ID nằm trong `<seq>/switches_oracle.csv`, từng cửa sổ trong `<seq>/smp_windows_oracle.csv`. `base_idsw − switches_fixed + switches_introduced` chỉ xấp xỉ `smp_idsw`: một lỗi có thể bị dời hoặc tách thành hai lỗi gần nhau trên cùng đối tượng.

Chạy với detector thật (GPU):

```bash
python scripts/extract_frames.py --data datasets/FastTracker/train        # chỉ cần nếu sequence không có img1/
python scripts/detect_yolo.py --data datasets/FastTracker/train --weights yolo11x.pt --out outputs/dets
python scripts/run_pilot.py --data datasets/FastTracker/train --dets dir --det-dir outputs/dets --backend oracle
```

Kiểm tra SAM 3 chạy được trên GPU (đo ms/khung và VRAM):

```bash
python scripts/smoke_sam3.py --data datasets/FastTracker/train --frame 100 --objects 5 --len 60
```

### Tuần 6–11: Risk-Aware Dispatch

```bash
# nhãn y_risk (chỉ cần GT), quét chính sách với VOS oracle
python scripts/run_dispatch.py --data datasets/FastTracker/train --dets oracle --backend oracle --label y_risk --out outputs/dispatch
# nhãn y_fix (chạy thử cửa sổ để biết cửa sổ nào thật sự sửa được lỗi)
python scripts/run_dispatch.py ... --label y_fix --fix-backend oracle
# xác nhận với SAM 3 thật (GPU), có thể giới hạn vài sequence bằng --seqs
python scripts/run_dispatch.py ... --backend sam3 --seqs <seq1> <seq2>
```

Kết quả:
- `model_metrics_*.json`: ROC-AUC, PR-AUC của mô hình so với dùng margin đơn thuần;
- `sweep_summary.csv`: chi phí (số cửa sổ, số khung VOS) so với IDSW, IDF1 cho từng chính sách;
- `tradeoff_*.png`: đường cong chi phí theo độ chính xác.

Các chính sách được so sánh:
- `smp`: ngưỡng margin của bài gốc;
- `smp+birth`: thêm ứng viên "birth" cho lỗi đứt track;
- `density`: quy tắc theo mật độ;
- `learned`: top-k theo xác suất rủi ro, với ngân sách bằng k lần số trigger của SMP;
- `random`: chọn ngẫu nhiên cùng ngân sách.

## Những điều cần ghi rõ khi báo cáo

1. **SMP là bản cài lại.** Bài gốc không công bố mã nguồn và phần lớn siêu tham số. Các giá trị đánh dấu (*) trong `configs/default.yaml` là do mình chọn. Nên kiểm tra trên DanceTrack val trước (bài gốc báo SAM 3 + Deep-EIoU tăng 8,0 HOTA). Dữ liệu tải tại https://github.com/DanceTrack/DanceTrack.
2. **Khác biệt so với bài gốc,** cần nêu trong báo cáo:
   - `tau_box_cov`: mask phải lấp đầy một phần box mới được tính là "nằm trong" track. Thêm vào để tránh ghi đè sai khi vật bị che.
   - Kết cục `LOST`: mask rỗng kéo dài, tức đối tượng bị che lâu.
   - Khung hoán đổi được xác định là khung đầu tiên sau lần cuối mask còn nằm trong track được mồi ít nhất bằng trong track mới.
   - Hai cửa sổ cùng mô tả một sự kiện (A→B và B→A, hoặc hai cửa sổ chồng thời gian trên cùng cặp) chỉ được áp dụng một lần.
   - Witness được chạy thành các cửa sổ riêng; bài gốc không nói rõ cách làm.
3. **Visibility của FastTracker là hằng số 1.0.** Code ước lượng che khuất từ hình học box (`rad/occlusion.py`, box nằm thấp hơn trong ảnh được coi là gần camera hơn) cho detection oracle và VOS oracle. Đây là xấp xỉ, cần nêu rõ khi báo cáo.
4. **VOS oracle là cận trên,** không phải kết quả của SAM. Kết luận cuối cùng phải xác nhận lại bằng `--backend sam3`.
5. **Dữ liệu giả lập** (`scripts/make_synthetic.py`) chỉ dùng để kiểm tra mã chạy đúng. Không đưa số liệu từ dữ liệu này vào báo cáo như bằng chứng.

## Kiểm tra nhanh mã nguồn

```bash
python -m pytest -q
python scripts/make_synthetic.py --out datasets/SYNTH/train --n 6
python scripts/run_pilot.py --data datasets/SYNTH/train --dets det --backend oracle --out outputs/pilot_synth
```

## Việc còn để ngỏ

- Backend SAM 3.1 (Object Multiplex) dùng API phiên khác với SAM 3 tracker. Đường mồi bằng box cho một đối tượng cụ thể cần kiểm tra thêm (problem P5).
- Fine-tune YOLO trên FastTracker (README của FastTracker hướng dẫn với YOLOX).
- Cửa sổ witness chạy chung một lần SAM.
