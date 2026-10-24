# Hướng dẫn chạy thí nghiệm trên Kaggle từ đầu

Hướng dẫn này dành cho lần đầu dùng Kaggle. Script nhận một hoặc nhiều phương pháp qua `--methods`; bỏ tham số đó sẽ chạy cả sáu phương pháp tuần tự. Mặc định dùng toàn bộ hộ đủ điều kiện, 3 communication rounds, 3 local epochs, batch size 64 và không thêm synthetic drift.

## Trước khi bắt đầu

- Mỗi Kaggle notebook CPU/GPU có giới hạn phiên tối đa 12 giờ. Hạn mức GPU mỗi tuần thường khoảng 30 giờ, có thể thay đổi theo nhu cầu và tài nguyên. Kiểm tra quota GPU trong trang Kaggle trước khi khởi chạy.
- Bản script có checkpoint sau mỗi phương pháp. Nếu một phương pháp lỗi Python, các phương pháp đã hoàn tất được giữ lại; khi tiếp tục, script bỏ qua chúng và chạy lại từ phương pháp bị lỗi. Nếu phiên bị Kaggle cưỡng bức dừng giữa phương pháp, riêng phương pháp đang chạy vẫn phải làm lại từ đầu.
- Script tự chọn `cuda:0` khi CUDA khả dụng. GPU giúp phần PyTorch; đọc các CSV và tiền xử lý pandas vẫn chạy CPU. Nếu Kaggle cấp hai GPU, bản script hiện tại chỉ dùng GPU đầu tiên.
- Không chạy sáu tài khoản do cùng một người điều khiển. Điều khoản Kaggle chỉ cho phép một tài khoản hoạt động do một người sử dụng. Nếu có sáu cộng tác viên thật, mỗi người tự dùng tài khoản của mình; có thể mời họ vào private dataset. Nếu làm một mình, dùng một tài khoản và chỉ chạy đồng thời trong phạm vi session/quota Kaggle cho phép.

## Bước 1: Tạo tài khoản và bật quyền GPU

1. Mở [Kaggle](https://www.kaggle.com/) và đăng ký/đăng nhập bằng tài khoản của bạn.
2. Nếu mục GPU bị mờ hoặc không xuất hiện, vào trang tài khoản Kaggle và hoàn thành Phone Verification.
3. Không gửi mật khẩu hoặc token API Kaggle cho cộng tác viên. Mỗi người cần tự đăng nhập tài khoản của mình.

## Bước 2: Chuẩn bị bộ dữ liệu tải lên

Trong Kaggle, dữ liệu sẽ được gắn dưới `/kaggle/input` ở chế độ chỉ đọc. File kết quả phải ghi vào `/kaggle/working`.

Hãy tạo một thư mục upload riêng, chỉ có cấu trúc sau; không chép `venv`, `results`, `archive` hay log local vào đó:

```text
kaggle_upload/
  run_all_blocks.py
  merge_method_results.py
  data/
    london/
      informations_households.csv
      halfhourly_dataset/
        block_0.csv
        ...
        block_111.csv
```

Hai file Python nằm trong thư mục gốc của dự án là `run_all_blocks.py` và `merge_method_results.py`; dữ liệu nguồn ở `data/london`. Có thể tải folder `kaggle_upload` trực tiếp, hoặc nén folder thành ZIP. ZIP thường tiện hơn với bộ dữ liệu lớn/nhiều file; Kaggle tự giải nén archive khi tạo dataset. Tổng 112 block tại máy hiện khoảng 7.33 GB. Giới hạn dataset Kaggle hiện lớn hơn kích thước này, nhưng phần private quota của tài khoản vẫn cần còn đủ chỗ.

## Bước 3: Tạo private Dataset

1. Trên Kaggle, mở **Datasets** rồi chọn **New Dataset**.
2. Tải folder `kaggle_upload` hoặc ZIP vừa tạo.
3. Đặt tên, ví dụ `deco-london-input`, và để **Private**.
4. Chọn **Create Dataset** và chờ Kaggle xử lý xong. Với nhiều GB dữ liệu, xử lý có thể mất vài phút hoặc lâu hơn.
5. Nếu làm với cộng tác viên, mở **Settings → Sharing** của dataset, tìm username Kaggle của họ và cấp quyền **Can view**.

Private dataset mặc định chỉ hiển thị với chủ sở hữu và cộng tác viên được mời. Không đổi sang Public nếu chưa chắc quyền chia sẻ dữ liệu cho phép.

## Bước 4: Tạo Notebook và gắn Dataset

1. Chọn **Create → New Notebook**; giữ ngôn ngữ Python.
2. Trong khung **Input** bên phải, bấm **Add Input** và thêm dataset London. Nếu đã tạo dataset code-only có `run_all_blocks_fixed.py`, hãy thêm dataset đó nữa; bản này có sửa lỗi Fisher và hỗ trợ resume.
3. Trong **Session options** hoặc **Settings**, đặt **Accelerator → GPU**. Việc chuyển accelerator có thể khởi động lại session.
4. Nếu chọn GPU thành công, chạy cell sau để xác nhận. Chưa chạy thí nghiệm nếu `CUDA` là `False`.

```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

GPU miễn phí có thể phải đợi nếu đang đông người. Notebook Kaggle đã cài sẵn PyTorch và nhiều thư viện khoa học dữ liệu thường dùng.

### Cập nhật riêng script mà không tải lại dữ liệu lớn

Nếu cần cập nhật code sau khi đã tải dataset 7.33 GB, có thể tạo một private Dataset nhỏ chỉ chứa file script mới, rồi gắn thêm dataset đó bằng **Add Input**. Trong cell chạy, tìm đúng tên script mới thay vì tìm `run_all_blocks.py`. Ví dụ file hotfix tên `run_all_blocks_fixed.py` thì dùng:

```python
scripts = list(Path('/kaggle/input').rglob('run_all_blocks_fixed.py'))
```

Đường dẫn dữ liệu vẫn được tìm từ dataset London đang gắn riêng.

## Bước 5: Tìm đường dẫn dữ liệu và script

Chạy cell này. Nó tự tìm folder, không cần biết slug Kaggle tự sinh:

```python
from pathlib import Path

input_root = Path('/kaggle/input')
data_dirs = [
    p.parent for p in input_root.rglob('informations_households.csv')
    if (p.parent / 'halfhourly_dataset').is_dir()
]
scripts = list(input_root.rglob('run_all_blocks_fixed.py'))
if not scripts:
    scripts = list(input_root.rglob('run_all_blocks.py'))

print('London data:', data_dirs)
print('Script:', scripts)
assert len(data_dirs) == 1, 'Cần đúng một London data input có đủ 112 block.'
assert len(scripts) == 1, 'Cần đúng một script chạy (ưu tiên run_all_blocks_fixed.py).'

data_dir = data_dirs[0]
script = scripts[0]
blocks = list((data_dir / 'halfhourly_dataset').glob('block_*.csv'))
print('Số block:', len(blocks))
assert len(blocks) == 112, f'Mong đợi 112 block, tìm thấy {len(blocks)}.'
```

Nếu assert không qua, dừng ở đây và xem cấu trúc file trong tab **Input**. Script cần `informations_households.csv` và thư mục `halfhourly_dataset` nằm cạnh nhau.

## Bước 6: Chạy thử để xác nhận thiết lập

Chạy thử Local-FT trên tối đa hai hộ mỗi Acorn group (tối đa 6 hộ). Bước này vẫn đọc các block để lọc hộ nên có thể mất thời gian tải/đọc dữ liệu, nhưng phần train nhỏ. Kết quả thử nghiệm này **không** phải kết quả nghiên cứu cuối cùng.

```python
import subprocess
import sys
from pathlib import Path

pilot_dir = Path('/kaggle/working/pilot')
subprocess.run([
    sys.executable, str(script),
    '--data-dir', str(data_dir),
    '--output-dir', str(pilot_dir),
    '--max-clients-per-group', '2',
    '--methods', 'Local-FT',
], check=True)
print(list(pilot_dir.iterdir()))
```

Cell chạy đến cuối không báo lỗi là đã xác nhận được đường dẫn, thư viện và GPU. Có thể kiểm tra file `run_config.json` để thấy số hộ và thiết bị. Sau đó bỏ `--max-clients-per-group 2` ở lượt chính thức.

## Bước 7: Chạy cả sáu phương pháp trong một lượt

Bạn muốn chạy toàn bộ 5,057 hộ cho cả sáu phương pháp trong cùng một lần bấm. Script sẽ chạy phương pháp này xong rồi mới chuyển sang phương pháp tiếp theo; chúng không chạy đồng thời. Không truyền `--methods`, vì mặc định script chạy cả sáu:

```text
Local-FT
Dec-Avg
Dec-EWC
Dec-Replay
SimDec-FT
DeCo-LSTM
```

Thay cell chạy thử bằng cell sau. Đoạn này tự tìm dữ liệu/script và checkpoint nếu bạn đã thêm output của một lượt trước vào **Input**. Lần chạy đầu tiên không cần thêm output cũ.

```python
from pathlib import Path
import subprocess
import sys
import torch

input_root = Path('/kaggle/input')
data_dirs = [
    p.parent for p in input_root.rglob('informations_households.csv')
    if (p.parent / 'halfhourly_dataset').is_dir()
]
scripts = list(input_root.rglob('run_all_blocks_fixed.py'))
if not scripts:
    scripts = list(input_root.rglob('run_all_blocks.py'))
assert len(data_dirs) == 1, f'Cần đúng một London data input; tìm thấy {len(data_dirs)}.'
assert len(scripts) == 1, f'Cần đúng một script chạy; tìm thấy {len(scripts)}.'

data_dir = data_dirs[0]
script = scripts[0]
blocks = list((data_dir / 'halfhourly_dataset').glob('block_*.csv'))
assert len(blocks) == 112, f'Mong đợi 112 block; tìm thấy {len(blocks)}.'
assert torch.cuda.is_available(), 'GPU chưa bật hoặc CUDA không khả dụng.'

output_dir = Path('/kaggle/working/results_all_methods')
resume_files = list(input_root.rglob('table1_partial_results.csv'))
assert len(resume_files) <= 1, (
    'Tìm thấy nhiều checkpoint. Hãy bỏ các output cũ không định resume khỏi Input, '
    'chỉ giữ output của lượt chạy muốn tiếp tục.'
)

command = [
    sys.executable, str(script),
    '--data-dir', str(data_dir),
    '--output-dir', str(output_dir),
    '--comm-rounds', '3',
    '--local-epochs', '3',
    '--batch-size', '64',
]
if resume_files:
    resume_dir = resume_files[0].parent
    print('Resume checkpoint:', resume_dir)
    command += ['--resume-from', str(resume_dir)]
else:
    print('Không có checkpoint trong Input; bắt đầu lượt mới.')

subprocess.run(command, check=True)
status_path = output_dir / 'run_status.json'
if status_path.exists():
    print('Run status:', status_path.read_text())
```

Không truyền `--max-clients-per-group` để dùng tất cả hộ; không truyền `--inject-synthetic-drift`. Đầu log cần hiện `Device: cuda`; nếu là `cpu`, hãy dừng và kiểm tra GPU trong Session options. Nếu đã thêm output cũ vào Input, cell tự tìm `table1_partial_results.csv` và nối `--resume-from`; tránh gắn nhiều output checkpoint cùng lúc.

### Lưu và để lượt chạy hoàn tất

Với lượt chạy dài, chọn **Save Version → Save & Run All** một lần để Kaggle chạy notebook trong phiên riêng. Nếu cell pilot còn trong notebook, hãy xóa hoặc bỏ qua cell đó trước lượt chính thức. Kaggle yêu cầu toàn notebook hoàn tất trong tối đa 12 giờ. Local-FT đã mất khoảng 20.8 phút lần trước; tổng thời gian còn phụ thuộc các phương pháp tiếp theo. Khi script bắt được lỗi Python trong một phương pháp, nó lưu checkpoint rồi kết thúc cell bình thường để Kaggle lưu thư mục output. Hãy mở `run_status.json`: `paused_after_error` nghĩa là cần resume; `complete` nghĩa là đủ các phương pháp được chọn. Nếu Kaggle cưỡng bức dừng phiên vì chạm giới hạn thời gian hoặc tài nguyên, không thể trông cậy vào output của phiên đó; chỉ resume được từ checkpoint đã lưu thành Output. Đóng trình duyệt hoặc cho máy tính sleep không dừng phiên Kaggle đã lưu.

Mỗi phương pháp được ghi vào cùng `results_all_methods`; file ma trận của phương pháp được ghi trước `table1_partial_results.csv`, bảng này đóng vai trò mốc commit sau khi phương pháp hoàn tất. `run_status.json` lưu trạng thái; nếu lỗi Python, `run_error.txt` lưu traceback. Resume kiểm tra cấu hình/số hộ và bỏ qua phương pháp đã có cả dòng kết quả lẫn file ma trận. Nếu lỗi nằm giữa một phương pháp, chỉ phương pháp đó phải chạy lại từ đầu. GPU Kaggle hiển thị T4 x2 nhưng script hiện chỉ dùng `cuda:0`, và sáu phương pháp vẫn chạy tuần tự. Mỗi job đọc/tiền xử lý 112 block một lần khi bắt đầu.

## Bước 8: Tải kết quả về

Output folder `results_all_methods` có `table1_main_results.csv`, `table1_partial_results.csv`, `run_status.json`, `run_signature.json`, các ma trận `*_R_rmse.npy`, `R_rmse_by_method.npy`, `client_selection.csv`, `run_config.json`; nếu một phương pháp lỗi còn có `run_error.txt`. DeCo-LSTM còn có similarity matrix và hình.

Có thể tạo một ZIP kết quả ngay sau khi lệnh train hoàn tất:

```python
import shutil
archive_path = shutil.make_archive(
    '/kaggle/working/all_methods_results', 'zip', root_dir=output_dir
)
print(archive_path)
```

Trong tab **Output**, tải ZIP này về máy.

## Bước 9: Xác nhận bảng tổng hợp

Với lượt chạy cả sáu phương pháp, không cần chạy `merge_method_results.py`. Mở `table1_main_results.csv` trong ZIP tải về và xác nhận có đủ sáu dòng, từ `Local-FT` đến `DeCo-LSTM`. `run_config.json` ghi lại số hộ, accelerator và danh sách phương pháp đã chạy. Chỉ dùng `merge_method_results.py` nếu bạn đã chạy từng phương pháp thành các phiên riêng.

## Lỗi thường gặp

- **Không thấy GPU / `CUDA: False`:** kiểm tra Phone Verification, chọn GPU trong Session options rồi chờ session khởi động lại.
- **Không thấy dữ liệu:** dataset chưa được thêm bằng **Add Input**, hoặc cấu trúc không có `informations_households.csv` cạnh `halfhourly_dataset`.
- **Tìm thấy ít hơn 112 block:** chưa tải đủ dữ liệu hoặc giải nén sai cấu trúc; không chạy kết quả cuối cùng cho đến khi đủ 112.
- **`ModuleNotFoundError`:** đọc tên package trong lỗi. Nếu cần cài package bằng `pip`, Internet phải bật trong session lúc cài; sau đó có thể tắt Internet khi chạy nếu mọi package đã sẵn có.
- **`RuntimeError: cudnn RNN backward can only be called in training mode`:** dùng `run_all_blocks_fixed.py` bản mới trong code-only dataset. Không cần tải lại dataset London.
- **Script báo `paused_after_error`:** tải hoặc thêm Output của version vừa chạy vào Input notebook, giữ lại một checkpoint `table1_partial_results.csv`, rồi chạy lại cell chính. Script sẽ tự truyền `--resume-from` và bỏ qua các phương pháp đã hoàn thành.
- **Resume báo thiếu file ma trận hoặc không khớp cấu hình:** giữ nguyên output/checkpoint đầy đủ và dùng cùng số rounds, epochs, batch size, danh sách phương pháp, tập hộ và cài đặt drift như lượt trước. Không sửa hoặc xóa riêng file trong checkpoint.
- **Notebook chạm giới hạn 12 giờ:** chỉ có thể tiếp tục từ output đã được lưu thành version; phương pháp chưa hoàn tất phải chạy lại từ đầu. Nếu version trước còn ở trạng thái đang chạy hoặc bị cưỡng bức dừng, chờ nó dừng hẳn rồi kiểm tra Output trước khi tạo job resume.
- **Output không tải được:** xác nhận notebook chạy xong, sau đó mở tab **Output** và tải file ZIP trong `/kaggle/working`.

## Thực nghiệm ablation và kiểm định độ ổn định

Khi cần so sánh từng thành phần của DeCo-LSTM với cấu hình khớp hơn và chạy nhiều seed, dùng [ABLATION_STUDY.md](ABLATION_STUDY.md). Script ablation là `run_ablation.py`; không thay thế lượt benchmark sáu phương pháp đã hoàn thành.
