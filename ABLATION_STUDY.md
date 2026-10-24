# Thí nghiệm ablation và baseline khớp cấu hình

Tài liệu này hướng dẫn chạy các thí nghiệm để xác định thành phần nào đóng góp vào kết quả của DeCo-LTSM, đồng thời so sánh với các baseline được chạy trong cùng điều kiện.

## 1. Thiết lập thí nghiệm

Các lần chạy dùng chung tập dữ liệu London đã lọc còn **5.057 hộ đạt chuẩn**, gồm **112 block**, cùng cách chia dữ liệu theo mùa và cấu hình huấn luyện:

- 3 vòng giao tiếp liên thiết bị (`comm-rounds=3`)
- 3 epoch cục bộ mỗi vòng (`local-epochs=3`)
- batch size 64
- tối đa 8 peer cho thành phần similarity
- Các biến thể dùng chung dữ liệu, khởi tạo mô hình và quy trình đánh giá trong cùng seed

Mỗi mẫu dùng 48 điểm đo nửa giờ để dự báo điểm đo nửa giờ kế tiếp. Dữ liệu được chuẩn hóa theo trung bình và độ lệch chuẩn mùa Winter của từng hộ. Vì vậy các chỉ số lỗi trong kết quả là **đơn vị đã chuẩn hóa**, không phải kWh.

## 2. Các biến thể cần chạy

| Tên biến thể | Thay đổi so với DeCo đầy đủ | Mục đích |
|---|---|---|
| `DeCo-Full` | Đầy đủ các thành phần DeCo-LTSM | Mốc chính để đối chiếu |
| `DeCo-noEWC` | Tắt phần phạt EWC; vẫn tính Fisher nếu cần cho cổng bộ nhớ | Đo đóng góp của EWC |
| `DeCo-noReplay` | Tắt replay | Đo đóng góp của replay |
| `DeCo-noSimilarity` | Thay đồ thị peer học từ similarity bằng 8 peer ngẫu nhiên, trọng số peer đồng đều | Đo đóng góp của lựa chọn peer theo similarity |
| `DeCo-noGate` | Tắt cổng động; giữ tỷ lệ lưu giữ cố định 0,65 cho feature extractor và 0,88 cho head | Đo đóng góp của cổng động |
| `Matched-EWC` | Chỉ dùng EWC, với cường độ EWC của DeCo | Baseline EWC khớp cấu hình |
| `Matched-Replay` | Chỉ dùng replay, với buffer và top-k của DeCo | Baseline replay khớp cấu hình |
| `Matched-Similarity` | Chỉ dùng similarity, với tối đa 8 peer như DeCo | Baseline similarity khớp cấu hình |

Các baseline khớp cấu hình giúp so sánh từng thành phần riêng với DeCo đầy đủ trong cùng điều kiện huấn luyện. Mỗi lần chạy gồm bốn biến thể để phù hợp hơn với thời lượng một phiên Kaggle.

## 3. Chuẩn bị Kaggle

1. Tải file `run_ablation.py` trong thư mục `kaggle_patch` lên Kaggle dưới dạng một **private Code dataset**. Đây là file mã chạy thí nghiệm, không phải notebook.
2. Tạo notebook Kaggle, bật GPU và đính kèm hai input:
   - Dataset London hiện có, chứa `data/london`.
   - Code dataset mới chứa `run_ablation.py`.
3. Dùng cell bên dưới để tự tìm đường dẫn dữ liệu và mã chạy. Cell kiểm tra tập dữ liệu có đủ 112 block và GPU CUDA đã sẵn sàng.

```python
from pathlib import Path
import json
import shutil
import subprocess
import sys
import torch

input_root = Path('/kaggle/input')
data_dirs = [
    p.parent for p in input_root.rglob('informations_households.csv')
    if (p.parent / 'halfhourly_dataset').is_dir()
]
scripts = list(input_root.rglob('run_ablation.py'))
assert len(data_dirs) == 1, f'Tìm thấy {len(data_dirs)} bộ dữ liệu London.'
assert len(scripts) == 1, f'Tìm thấy {len(scripts)} file run_ablation.py.'
assert torch.cuda.is_available(), 'GPU/CUDA chưa khả dụng.'

data_dir = data_dirs[0]
script = scripts[0]
assert len(list((data_dir / 'halfhourly_dataset').glob('block_*.csv'))) == 112

# Đổi hai giá trị này trước mỗi lần Save & Run All:
SEED = 42
BATCH = 1
METHODS_BY_BATCH = {
    1: ['DeCo-Full', 'DeCo-noEWC', 'DeCo-noReplay', 'DeCo-noSimilarity'],
    2: ['DeCo-noGate', 'Matched-EWC', 'Matched-Replay', 'Matched-Similarity'],
}
methods = METHODS_BY_BATCH[BATCH]
output_dir = Path(f'/kaggle/working/ablation_seed_{SEED}_batch{BATCH}')

# Nếu input có kết quả của lần chạy lỗi trước, tìm checkpoint đúng seed và batch.
resume_dirs = []
for partial in input_root.rglob('table1_partial_results.csv'):
    signature_path = partial.parent / 'run_signature.json'
    if not signature_path.is_file():
        continue
    signature = json.loads(signature_path.read_text())
    if (
        signature.get('experiment_version') == 'deco_ablation_v1'
        and signature.get('seed') == SEED
        and signature.get('methods') == methods
    ):
        resume_dirs.append(partial.parent)
assert len(resume_dirs) <= 1, f'Tìm thấy nhiều checkpoint khớp: {resume_dirs}'

command = [
    sys.executable, str(script),
    '--data-dir', str(data_dir),
    '--output-dir', str(output_dir),
    '--seed', str(SEED),
    '--comm-rounds', '3',
    '--local-epochs', '3',
    '--batch-size', '64',
    '--methods', *methods,
]
if resume_dirs:
    command += ['--resume-from', str(resume_dirs[0])]
    print('Tiếp tục từ checkpoint:', resume_dirs[0])
else:
    print('Bắt đầu batch mới:', SEED, BATCH, methods)

subprocess.run(command, check=True)
status_path = output_dir / 'run_status.json'
print(status_path.read_text() if status_path.exists() else 'Không thấy run_status.json.')
archive_path = shutil.make_archive(str(output_dir), 'zip', root_dir=str(output_dir))
print('File ZIP:', archive_path)
```

## 4. Chạy nhiều seed và tiếp tục khi bị ngắt

Để có kết quả đáng tin cậy hơn một lần chạy, chạy cả hai batch với ba seed: `42`, `123`, `2026`. Tổng cộng là **6 lần Save & Run All**:

| Seed | Lần chạy thứ nhất | Lần chạy thứ hai |
|---:|---|---|
| 42 | Đặt `BATCH = 1` | Đặt `BATCH = 2` |
| 123 | Đặt `BATCH = 1` | Đặt `BATCH = 2` |
| 2026 | Đặt `BATCH = 1` | Đặt `BATCH = 2` |

Trước mỗi lần chạy, sửa `SEED` và `BATCH` trong cell rồi chọn **Save & Run All**. Mỗi seed/batch có thư mục kết quả riêng trong `/kaggle/working/` và được nén thành file ZIP.

Nếu phiên Kaggle bị ngắt do lỗi hoặc hết thời gian:

1. Tải Output của phiên bị ngắt xuống máy hoặc lưu lại theo cách Kaggle cho phép.
2. Đính kèm Output đó làm Input cho notebook chạy tiếp.
3. Giữ nguyên `SEED` và `BATCH` của lần chạy đó rồi chạy lại cell.
4. Cell sẽ tìm checkpoint có cùng seed, batch và phiên bản thí nghiệm. Script tiếp tục từ ranh giới phương pháp đã lưu.

Nếu lỗi xảy ra giữa lúc đang huấn luyện một phương pháp, phương pháp đó có thể cần chạy lại từ đầu; các phương pháp đã hoàn tất và lưu checkpoint sẽ được giữ lại. Kaggle có thể đánh dấu phiên là hoàn tất nếu lỗi đã được chương trình ghi nhận; hãy kiểm tra `run_status.json` để biết trạng thái thật. Chỉ coi một batch là hoàn tất khi trạng thái cho biết đã xong và có đủ kết quả cho bốn phương pháp.

## 5. Tổng hợp kết quả

Sau khi tải xuống và giải nén đủ sáu thư mục kết quả, đặt chúng cạnh `analyze_ablation.py` trong thư mục dự án rồi chạy trong PowerShell:

```powershell
.\venv\Scripts\python.exe .\analyze_ablation.py `
  --seed-dirs `
    .\ablation_seed_42_batch1 .\ablation_seed_42_batch2 `
    .\ablation_seed_123_batch1 .\ablation_seed_123_batch2 `
    .\ablation_seed_2026_batch1 .\ablation_seed_2026_batch2 `
  --output-dir .\ablation_analysis
```

Chương trình tạo thư mục `ablation_analysis` với hai bảng:

- `ablation_summary.csv`: trung bình và độ lệch chuẩn của các chỉ số qua những seed đã cung cấp.
- `paired_block_bootstrap.csv`: khoảng tin cậy bootstrap ghép cặp theo seed và block, so sánh `DeCo-Full` với từng biến thể.

Trong bảng so sánh, chênh lệch được tính theo `DeCo-Full − biến thể`. Với AE-RMSE và AE-MAE, chênh lệch âm nghĩa là DeCo-Full có lỗi thấp hơn. Với BWT, giá trị âm hơn thể hiện mức lỗi trên các mùa trước giảm nhiều hơn theo định nghĩa đang dùng trong mã.

## 6. Ước tính thời gian

Lần chạy đủ sáu phương pháp trước đó mất khoảng **418,79 phút** trên Kaggle GPU T4. Tám biến thể ablation chạy theo ba seed tương ứng 24 lượt phương pháp, nên có thể cần khoảng **20–30 giờ GPU tổng cộng**; thời gian thực tế thay đổi theo tải máy, kích thước batch xử lý và thời lượng phiên Kaggle. Kiểm tra hạn mức GPU của tài khoản trước khi khởi chạy cả sáu batch.

`run_ablation.py` mới được kiểm tra cú pháp tĩnh; chưa được chạy toàn bộ trên Kaggle. Vì vậy, hãy chạy batch đầu tiên và kiểm tra `run_status.json`, các file CSV và ZIP trước khi gửi các batch còn lại.
