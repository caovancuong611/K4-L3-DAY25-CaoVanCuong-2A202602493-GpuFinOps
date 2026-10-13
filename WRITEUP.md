# Lab 25 — GPU FinOps: Bài viết phân tích

**Học viên:** Cao Văn Cường — 2A202602493
**Kết quả kiểm tra:** `python verify.py` → 11/11 checks passed · `pytest -q` → 25 passed (15 test gốc + 10 test tự viết cho extension, không sửa test gốc)
**Đầu ra:** `outputs/report.md`, `outputs/savings.png`, `outputs/focus_export.csv`

Mọi con số dưới đây lấy từ output của `python missions/run_all.py` (dữ liệu seed=25), trùng với `outputs/report.md`.

---

## 1. Baseline và Optimized

| | Baseline | Optimized | Tiết kiệm |
|---|---|---|---|
| Tổng chi phí GPU / tháng | $27,133 | $15,004 | $12,129 (**45%**) |
| Inference API / ngày | $48.87 | $8.88 | 81.8% |
| **$/1M-token** | **$6.488** | **$1.179** | rẻ hơn 5.5 lần |
| Mua GPU (workloads) / tháng | $25,667 (on-demand) | $15,878 | 38.1% |

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | $/tháng | % baseline |
|---|---|---|
| Purchasing (spot/reserved) | $9,789 | 36.1% |
| Inference (cascade/cache/batch) | $1,200 | 4.4% |
| Kill idle GPUs | $600 | 2.2% |
| Right-size (MBU/roofline) | $540 | 2.0% |

- **Theo số tiền, Purchasing lớn nhất**, vì chi phí training/inference trên GPU thuê (~$25.7k) lớn gấp ~17 lần hóa đơn API token (~$1.5k). Spot giảm ~37% cho các job có checkpoint, còn reserved 3 năm giảm ~40% cho dịch vụ chạy 24/7.
- **Theo đơn vị $/1M-token, cascade lớn nhất:** một mình nó cắt 76.5% chi phí inference, vì 80% request đủ dễ cho model nhỏ (rẻ hơn 15 lần ở input, 37.5 lần ở output). Cache (+1.6%) và batch (+3.7%) chỉ chiết khấu tiếp trên một hóa đơn đã nhỏ, nên đóng góp ít.
- Nếu chỉ đo bằng $/GPU-giờ thì không thấy được hiệu quả của cascade: GPU vẫn vậy, nhưng mỗi 1M token rẻ hơn 5.5 lần.

## 3. GPU-Util Lie

`gpu-h100-4` báo **GPU-Util 98.2% nhưng MFU chỉ 0.194**, trong khi các H100 khỏe cùng cụm đạt ~0.41. `gpu-a10g-1` cũng bị gắn cờ (util 96.9%, MFU 0.268).

**Cơ chế:** GPU-Util của `nvidia-smi` chỉ đo tỷ lệ thời gian mà *có ít nhất một kernel* đang chạy. Nó không đo số SM đang bận hay tensor core có được cấp đủ dữ liệu không. MBU của `gpu-h100-4` cũng thấp (0.207), nghĩa là GPU không bão hòa cả compute lẫn băng thông, tức là phần lớn thời gian nó đang **chờ**. Nguyên nhân thường gặp là dataloader chậm, copy host↔device, đồng bộ, batch nhỏ, hoặc overhead khi launch kernel.

**Tác động tài chính:** so với MFU của các H100 khỏe, khoảng **$955/tháng** tiền thuê `gpu-h100-4` (tổng **$1,208/tháng** cho cả 2 GPU bị gắn cờ) không sinh ra FLOPs hữu ích. Nên sửa phần mềm trước (profile bằng DCGM/Nsight, tăng batch, dùng CUDA graphs, async data loading); đổi phần cứng là phương án dự phòng.

## 4. Các phần mở rộng đã làm (5/5)

| # | Extension | File | Kết quả đo được | Insight |
|---|---|---|---|---|
| 1 | `recommend_tier()` dựa trên chi phí | `finops/pricing.py`, `missions/m3_purchasing.py` | Savings M3: 39.1% → **38.1%**. Ở vòng đời 1–2 năm: 29.7% | Policy cũ tính sai: reserved chỉ tính giờ dùng (thực tế phải trả 24/7) và luôn giả định gói 3 năm. Gói 3 năm chỉ thắng gói 1 năm khi dịch vụ sống ≥ ~2.1–2.25 năm. Ma trận GPU × giờ/ngày × interruptible cho điểm hòa vốn ~16h/ngày |
| 2 | Right-sizing theo MBU | `missions/m1_efficiency_audit.py` | **$540/tháng** (a10g-0 → L4, h100-4 → MI300X) | GPU rẻ nhất theo giờ (L4) lại đắt nhất theo $/TB/s ($2.67 so với MI300X $0.37). A100 không thể hạ cấp vì đang dùng ~68 GB HBM. a10g-1 sang L4 sẽ thành MBU 60.4%, vượt ngưỡng |
| 3 | `cache_is_worth_it()` | `finops/pricing.py`, `missions/m2_inference_levers.py` | Điểm hòa vốn: small **5.83**, large **1.94** lần đọc/giờ. Thực tế: 18.8 và 4.2. Tiết kiệm từ cache: $1.20 → **$0.80/ngày** sau khi trừ phí ghi + lưu | Tier small cần số lần đọc gấp ~3 lần vì phí lưu trữ tính theo token, còn giá input của nó rẻ hơn 15 lần. 10/191 nhóm giờ thấp điểm không đáng cache |
| 4 | Ngân sách Reasoning | `missions/m2_inference_levers.py`, `missions/m5_report.py` | Reasoning chiếm **8.4% traffic** nhưng **16.4% chi phí** và **94% năng lượng** (148 Wh/query so với 0.86) | Quy tắc "chỉ reasoning khi router đã chọn tier large" tiết kiệm 23.9 kWh/ngày. Cap ở 10% không có tác dụng vì traffic hiện tại đã dưới mức đó. Ở đây reasoning là bài toán năng lượng hơn là bài toán tiền |
| 5 | Carbon-aware scheduling | `missions/m6_carbon_aware.py`, `finops/sustainability.py` | Chuyển 5 job interruptible (4,312 kWh/tháng) sang europe-north1 giảm **1,509 kgCO2e/tháng (−92%)** và $129 tiền điện | Rẻ nhất: us-east-wa · Sạch nhất: europe-north1 · Cân bằng: us-east-wa. Vùng sạch xa người dùng 65–110 ms, nên chỉ chuyển job batch/training |

Insight quan trọng nhất: **phải sửa mô hình tính tiền trước khi tối ưu.** Ext 1 cho thấy policy gốc đã thổi phồng savings của reserved (bỏ qua việc phải trả 24/7 và giả định cam kết 3 năm). Ext 3 cho thấy savings của cache phải trừ chi phí ghi và lưu trữ. Con số đúng hơn thì thấp hơn, nhưng là con số an toàn để dùng khi ký hợp đồng.

## 5. Khuyến nghị cho NimbusAI (vai trò FinOps lead)

1. **Tuần 1 — Bật cascade routing và tự động tắt GPU idle.** Tiết kiệm ~$1,800/tháng, giảm $/1M-token 5.5 lần, rủi ro thấp và đảo ngược được. Đồng thời profile `gpu-h100-4` để lấy lại ~$955/tháng FLOPs bị lãng phí.
2. **Tháng 1 — Chuyển toàn bộ job interruptible sang spot có checkpoint** (~$5.9k/tháng), đặt chúng ở us-east-wa hoặc europe-north1 (−76% đến −92% carbon), và áp quy tắc reasoning theo tier.
3. **Sau khi fleet ổn định — Mua reserved cho inference 24/7** (chat, RAG, search). Chỉ ký gói 3 năm cho dịch vụ chắc chắn tồn tại ≥ 2.25 năm, còn lại ký 1 năm. Phải làm *sau* bước 1–2 để không cam kết mua công suất mà mình sắp cắt bỏ. Duy trì tag coverage ≥ 80% (hiện 92%) để chargeback theo team.

---

### Cách chạy lại

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python verify.py              # 11/11
pytest -q                     # 25 passed
python missions/run_all.py    # in đầy đủ bảng của các extension
```
