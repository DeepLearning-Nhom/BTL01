# BTL01 – Xấp xỉ hàm Gaussian 2 chiều bằng MLP (PyTorch)

Học sâu – Bài tập xấp xỉ hàm số dùng MLP. Xấp xỉ
`f(x) = 1/(2π√det C) · exp(−½ (x−μ)ᵀ C⁻¹ (x−μ))`, `μ=[0,0]`, `C=[[0.3,−0.5],[−0.5,2]]`,
trên miền `x1∈[−2,2]`, `x2∈[−4,4]`; khảo sát ảnh hưởng của số mẫu **A**, số nơ-ron **B**, số tầng ẩn **C**.

**Thành viên / nhiệm vụ / tỷ lệ đóng góp (điền trước khi nộp, tổng 100%)**

| Họ tên | MSSV | Nhiệm vụ | Đóng góp |
|---|---|---|---|
| … | … | … | …% |

## 1. Cài đặt (Python ≥ 3.9)
```bash
pip install -r requirements.txt        # gồm torch, numpy, pandas, scipy, matplotlib
```
Không phụ thuộc đường dẫn cá nhân; mọi đường dẫn tính theo thư mục chứa `run_all.py`.

## 2. Chạy lại thí nghiệm — **code chính là notebook `BTL01_MLP_Gaussian.ipynb`**
```bash
pip install -r requirements.txt
jupyter notebook BTL01_MLP_Gaussian.ipynb     # Kernel ▸ Restart & Run All
```
Notebook tự chứa (không phụ thuộc `src/`), chia theo mục và có giải thích từng bước:

| Mục | Nội dung | Lệnh/ô chạy |
|---|---|---|
| 0 | cấu hình (`QUICK=True` để chạy thử vài phút, `SEEDS`, `DEVICE`) | ô cấu hình |
| 1 | hàm Gaussian, sinh/lưu/chia dữ liệu, khảo sát, hình 3D | ô 1.1 |
| 2 | `MLP(nn.Module)` | ô mô hình |
| 3 | huấn luyện/validation/checkpoint/đánh giá bằng PyTorch + demo baseline | ô huấn luyện |
| 4 | nhóm A, A2 (đối chứng), B, B2 (đối chứng), C | ô 4.1 … 4.5 (chạy riêng từng nhóm được) |
| 5 | hình tổng hợp, chọn mô hình cuối theo validation, bề mặt xấp xỉ | ô 5 |

Dựng lại báo cáo/slide từ kết quả vừa chạy (cần `python-docx`, `python-pptx`):
```bash
python tools/build_report.py      # -> report/BaoCao_BTL01.docx
python tools/build_slides.py      # -> slides/Slide_BTL01.pptx
# PDF: mở bằng Word/PowerPoint -> Save as PDF, hoặc: soffice --headless --convert-to pdf <file>
```
(Tuỳ chọn) Bản dòng lệnh tương đương notebook: `python run_all.py --exp all --engine torch` (xem `run_all.py`, thư mục `src/`).

## 3. Cấu trúc thư mục
```
BTL01_MLP_Gaussian.ipynb   CODE CHÍNH (PyTorch) – chạy toàn bộ thí nghiệm
run_all.py            (tuỳ chọn) bản CLI tương đương notebook
src/data.py           hàm Gaussian, sinh/lưu/nạp dữ liệu, chia train/val/test
src/model.py          MLP(nn.Module): nn.Linear + kích hoạt, tuỳ chọn số tầng/nơ-ron
src/engine_torch.py   (bản .py tương đương) huấn luyện PyTorch: TensorDataset/DataLoader, MSELoss, Adam,
                      vòng lặp zero_grad→forward→loss→backward→step, train()/eval()/no_grad(), checkpoint
src/engine_numpy.py   engine tham chiếu NumPy (backprop viết tay) – kiểm chứng thuật toán
src/experiments.py    cấu hình baseline và các nhóm A/B/C/A2/B2, ghi log, bảng tổng hợp
src/plotting.py       toàn bộ hình
data/                 train.csv (pool 2000), val.csv (200), test.csv (100), seed=2026
results/              runs_*.csv (từng lần chạy), summary_*.csv (TB±std), logs/*.csv (loss theo epoch),
                      final_model.json, engine.txt (engine đã dùng), data_stats.csv
figures/              hình dùng trong báo cáo/slide
models/               trọng số tốt nhất theo validation của từng lần chạy (.pt hoặc .npz)
report/ , slides/     báo cáo (.docx/.pdf) và slide (.pptx/.pdf)
tools/                script dựng báo cáo/slide từ results/
```

## 4. Thiết lập thí nghiệm (baseline)
| Mục | Giá trị |
|---|---|
| Dữ liệu | seed 2026; 3 luồng RNG độc lập (SeedSequence.spawn) cho train/val/test; pool train 2000; val 200; test 100 |
| Tiền xử lý | x₁/2, x₂/4 (theo miền cố định, không dùng thống kê val/test → không rò rỉ); y giữ nguyên |
| Mô hình | MLP, kích hoạt tanh ở tầng ẩn, đầu ra tuyến tính; khởi tạo mặc định `nn.Linear` |
| Tối ưu | Adam, lr = 3e-3 (chọn trên validation trong {3e-3, 1e-2, 3e-2}), batch 64, MSELoss |
| Điều kiện dừng | tối đa 1000 epoch, early stopping patience = 100 trên val loss; checkpoint theo val loss thấp nhất |
| Seed | 0,1,2,3,4 (khởi tạo + thứ tự batch) → báo cáo TB ± độ lệch chuẩn |
| Quy tắc | mỗi nhóm chỉ đổi A / B / C; **test chỉ dùng đánh giá cuối**, không chọn siêu tham số |

Chỉ số: MSE (chính), MAE, RMSE. Nhóm A dùng A mẫu đầu của pool (tập con lồng nhau).

## 5. Lưu ý về engine đã dùng để sinh kết quả đi kèm
Xem `results/engine.txt`. Nếu ghi `numpy-reference`, số liệu/hình trong bản này được sinh bởi engine NumPy tham chiếu
(cùng kiến trúc, khởi tạo U(±1/√fan_in), Adam, lịch seed; gradient được kiểm tra bằng sai phân hữu hạn:
`python -c "from src import engine_numpy as e; print(e.gradient_check())"` → ~1e-8).
Để có kết quả chính thức bằng PyTorch: `pip install torch`, mở notebook và *Restart & Run All*,
sau đó dựng lại báo cáo/slide bằng `tools/build_*.py`. Số liệu thay đổi nhẹ theo engine/phiên bản; xu hướng
được kỳ vọng giữ nguyên nhưng **cần đọc lại phần nhận xét** sau khi chạy lại.
