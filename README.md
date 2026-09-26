# WM_test1 — 2D 액션 게임 월드모델

직접 만든 게임을 봇이 플레이한 기록(화면 + 키 입력)으로, 키 입력에 따라 다음 화면을 생성하는 **확산(diffusion) 월드모델**을 학습합니다.

```
게임 (Panda3D, Kenney 에셋)  →  봇 플레이 데이터 수집  →  서버에서 학습 (4×GPU)  →  로컬에서 모델 안에서 플레이
     play.py                      collect.py                 wm_server/train.py           play_model.py
```

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `wmgame/` | 게임 로직(`core.py`), 렌더러(`render.py`), 봇(`bot.py`), 녹화(`app.py`) |
| `play.py` | 직접 플레이 (녹화 가능) |
| `collect.py` | 봇으로 학습 데이터 수집 |
| `play_model.py` | 학습된 월드모델 안에서 플레이 |
| `setup_assets.py` | 에셋 다운로드 및 변환 |
| `wm_server/` | 학습 코드. **이 폴더만 서버로 옮기면 학습 가능** |

데이터(`wm_server/data/`), 에셋(`assets/`), 가중치(`*.pt`)는 용량 때문에 저장소에 포함하지 않습니다.

---

## 1. 설치 (게임 / 추론용 PC)

Python 3.10 이상 (3.13에서 확인).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate

# PyTorch (NVIDIA GPU, CUDA 12.6 빌드)
pip install torch --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
python setup_assets.py            # Kenney 에셋 다운로드 + 캐릭터 모델 변환 (최초 1회)
```

## 2. 게임 플레이

```bash
python play.py
python play.py --record           # 플레이를 학습 데이터 형식으로 data/human/ 에 저장
```

| 키 | 동작 |
|---|---|
| W A S D | 이동 |
| Space | 공격 |
| E | 포션 줍기 (체력 +3) |
| R | 재시작 |
| Esc | 종료 |

- 렌더링 320×240 (2배 슈퍼샘플링), 화면에는 960×720으로 확대 표시 (`--display 1280x960` 등으로 변경)
- 15 fps 고정 틱
- 몬스터: 스켈레톤(1방, 빠름) 60% / 좀비(2방, 느림) 40%, 최대 9마리
- 맵: 51×51 타일, 4×4 구역(집/창고/기둥 홀/폐허/바위/시장/목재 야적장), 시드마다 다름

## 3. 데이터 수집 (봇)

이번 데이터셋(약 20만 프레임, 3.9GB)은 아래 명령으로 만들었습니다.

```bash
# 일반 봇 + 매끄러운 봇(30%) — 몬스터 사냥, 허공 공격 가끔
python collect.py --episodes 330 --workers 4 --smooth-frac 0.3 --out wm_server/data/raw   # 190개에서 중단
# 공격 없이 가만히 서서 죽는 에피소드 (사망 장면용, 약 1만 프레임)
python collect.py --episodes 48 --passive --seed 20000 --workers 4 --out wm_server/data/raw

# 봇 플레이 구경 (저장 안 함)
python collect.py --episodes 1 --show --no-save
```

에피소드 하나 = `.npz` 파일 하나:

| 키 | 형태 | 설명 |
|---|---|---|
| `frames` | (T+1, 240, 320, 3) uint8 | 화면 |
| `actions` | (T, 6) uint8 | `w, a, s, d, attack, pickup` (멀티핫) |
| `hp`, `kills`, `damage`, `heal`, `reward`, `done`, `seed` | | 보조 정보 |

`frames[t]` 에서 `actions[t]` 를 누르면 `frames[t+1]` 이 됩니다.

## 4. 학습 (서버, 4×RTX 4090)

`wm_server/` 폴더를 데이터(`wm_server/data/raw/*.npz`)와 함께 서버로 복사합니다. 데이터는 저장소에 없으니 별도로 옮기세요 (예: `scp -r wm_server/data/raw user@server:~/WM_test1/wm_server/data/`).

```bash
cd wm_server
pip install -r requirements.txt

# 1) 에피소드들을 학습용 배열로 변환 (1회, 약 46GB 디스크 필요)
python prepare_data.py --raw data/raw --out data/packed

# 2) 학습
torchrun --nproc_per_node 4 train.py --data data/packed --out runs/wm
# GPU 1장: python train.py --data data/packed --out runs/wm
```

- 같은 명령을 다시 실행하면 `runs/wm/last.pt` 에서 이어서 학습합니다.
- 로그의 `samples/s` 로 전체 학습 시간을 가늠할 수 있습니다 (기본 15만 스텝).
- 5000 스텝마다 `runs/wm/samples/step_XXXXXX.png` 저장: 검증 에피소드에서 실제 키 입력으로 45프레임을 이어 생성한 결과 (각 쌍의 **위: 실제, 아래: 모델**).
- 메모리 부족 시 `--batch 8` (기본은 GPU당 16).
- 주요 옵션: `--steps`, `--lr`, `--context`(과거 프레임 수, 기본 4), `--channels`(기본 `64,64,128,256,256`), `--compile`.

결과물: `runs/wm/last.pt` (EMA 가중치 포함), `runs/wm/step_XXXXXX.pt` (2.5만 스텝마다 스냅샷).

## 5. 학습된 모델 실행 (로컬 PC)

서버의 `runs/wm/last.pt` 를 받아 `checkpoints/last.pt` 에 둡니다.

```bash
python play_model.py --ckpt checkpoints/last.pt --fp16
```

- 실제 게임으로 첫 4프레임을 만든 뒤, 그다음부터는 **모든 화면을 모델이 생성**합니다.
- 조작은 게임과 같고, `R` 을 누르면 새 시드의 실제 화면으로 다시 시작합니다.
- `--fp16`: RTX GPU의 텐서 코어 사용 (권장). `--steps 1` 이면 빠르지만 화질이 떨어질 수 있습니다.
- 창 제목에 생성 fps 가 표시됩니다.

예상 속도 (기본 모델, 디노이징 3스텝):

| GPU | 예상 fps |
|---|---|
| RTX 2060 (fp16) | 약 12~16 (추정) |
| MX250 (fp32) | 약 1.1 (실측) |

## 모델 요약

DIAMOND 방식의 EDM 확산 U-Net, 픽셀 공간에서 직접 생성 (약 3,200만 파라미터).

- 입력: 노이즈 섞인 다음 프레임 + 직전 4프레임 (15채널, 320×240)
- 조건: 노이즈 세기 + 직전 4틱의 키 입력 → 모든 블록의 정규화 층에 scale/shift로 주입
- U-Net: 320×240 → 160×120 → 80×60 → 40×30(어텐션) → 20×15(어텐션)
- 학습 시 과거 프레임에 노이즈를 섞어 자기 출력을 이어 받을 때의 붕괴를 완화
- 추론: Karras 스케줄 Euler 3스텝

한계: 과거 4프레임(약 0.27초)만 보므로, 화면 밖으로 나간 것은 기억하지 못합니다.

## 에셋

[Kenney](https://kenney.nl) Mini Dungeon, Graveyard Kit (CC0). `setup_assets.py` 가 내려받아 Panda3D에서 애니메이션이 동작하도록 변환합니다.
