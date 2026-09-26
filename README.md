# WM_test1 — 액션 게임 월드모델

![게임 플레이 화면](docs/hero.png)

**키보드 입력을 받아 다음 게임 화면을 "상상해서" 그려내는 신경망(월드모델)을 처음부터 끝까지 직접 만들어 보는 프로젝트**입니다.

게임 엔진 없이도 게임처럼 동작하는 모델을 만드는 것이 목표입니다. 이를 위해 작은 탑다운 액션 게임을 직접 만들고, 스크립트 봇과 사람이 그 게임을 플레이한 기록(화면 + 키 입력) 약 21만 프레임을 모아, 확산(diffusion) 모델이 **"지금까지의 화면 + 지금 누른 키 → 다음 화면"** 을 학습하게 합니다. 학습이 끝나면 실제 게임 대신 모델이 매 프레임을 생성하고, 사람은 그 안에서 WASD와 Space로 플레이할 수 있습니다.

```
게임 (Panda3D, Kenney 에셋)  →  봇 플레이 데이터 수집  →  서버에서 학습 (4×GPU)  →  로컬에서 모델 안에서 플레이
     play.py                      collect.py                 wm_server/train.py           play_model.py
```

### 게임

| 캐릭터 | 맵 (51×51 타일, 시드마다 생성) |
|---|---|
| ![캐릭터](docs/characters.png) | ![맵 전체](docs/map.png) |
| 왼쪽부터 스켈레톤(1방), 플레이어, 좀비(2방) | 모래색 길이 구역 사이를 격자로 잇고, 구역마다 묘지·묘실·집·창고·예배당·시장·숲 등을 배치 |

- **조작**: WASD 이동, Space 공격, E 포션 줍기 — 월드모델이 배울 행동은 이 6개 키뿐
- **규칙**: 체력 10, 몬스터가 다가와 공격, 공격하면 처치, 포션으로 회복
- **화면**: 320×240, 15 fps (학습 데이터와 같은 해상도로 렌더링)

### 학습 데이터 예시

봇이 기록한 실제 학습 프레임 (320×240, 3틱 간격). 스켈레톤이 다가오고 → 공격을 맞아 빨갛게 번쩍인 뒤 → 쓰러집니다. 모델은 이런 프레임과 그때 누른 키만 보고 다음 프레임을 예측하도록 학습합니다.

![학습 데이터 프레임](docs/frames.png)

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `wmgame/` | 게임 로직(`core.py`), 렌더러(`render.py`), 봇(`bot.py`), 녹화(`app.py`) |
| `play.py` | 직접 플레이 (녹화 가능) |
| `collect.py` | 봇으로 학습 데이터 수집 |
| `play_model.py` | 학습된 월드모델 안에서 플레이 |
| `setup_assets.py` | 에셋 다운로드 및 변환 |
| `tools/view_data.py` | 학습 데이터를 2×2 영상으로 확인 (키 입력·체력 표시) |
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
- 맵: 51×51 타일, 4×4 구역(묘지/묘실 마당/집/창고/예배당/폐허/바위 정원/시장/목재 야적장/숲), 시드마다 다름
- 이끼 초록 바닥 + 모래색 길(구역 사이 도로와 각 구역 입구), 밟고 지나가는 잔해·무덤 자리 장식

## 3. 데이터 수집 (봇)

현재 데이터셋(v2, 약 21만 프레임, 5.6GB):

| 구성 | 에피소드 | 프레임 | 비율 |
|---|---|---|---|
| 봇 (몬스터 사냥 + 가끔 허공 공격, 30%는 망설임 없는 매끄러운 봇) | 190 | 190,000 | 91% |
| 사람이 직접 플레이 (`play.py --record`, 사망 11판 포함) | 14 | 19,734 | 9% |

```bash
python collect.py --episodes 190 --workers 4 --smooth-frac 0.3 --out wm_server/data/raw
python play.py --record --out wm_server/data/raw     # 직접 플레이 (1분 = 900프레임)
# (선택) 공격 없이 가만히 서서 죽는 에피소드: collect.py --passive

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

# 1) 에피소드들을 학습용 배열로 변환 (1회, 약 48GB 디스크 필요)
python prepare_data.py --raw data/raw --out data/packed

# 2) 학습
torchrun --nproc_per_node 4 train.py --data data/packed --out runs/wm_v2
# GPU 1장: python train.py --data data/packed --out runs/wm_v2
```

- 같은 명령을 다시 실행하면 `<out>/last.pt` 에서 이어서 학습합니다. 설정(과거 프레임 수 등)을 바꿔 새로 학습할 때는 `--out` 폴더를 새로 지정하세요.
- 로그의 `samples/s` 로 전체 학습 시간을 가늠할 수 있습니다 (기본 15만 스텝).
- 5000 스텝마다 `<out>/samples/step_XXXXXX.png` 저장: 검증 에피소드에서 실제 키 입력으로 45프레임을 이어 생성한 결과 (각 쌍의 **위: 실제, 아래: 모델**).
- 메모리 부족 시 `--batch 8` (기본은 GPU당 16).
- 주요 옵션: `--steps`(기본 15만), `--lr`, `--context`(과거 프레임 수, 기본 8 = 약 0.5초), `--channels`(기본 `64,64,128,256,256`), `--compile`.

결과물: `runs/wm_v2/last.pt` (EMA 가중치 포함), `runs/wm_v2/step_XXXXXX.pt` (2.5만 스텝마다 스냅샷).

### 긴 롤아웃 비교 (재학습 없이 생성 설정만 바꿔 보기)

```bash
python compare_rollout.py
```

학습에 안 쓴 검증 에피소드를 자동으로 골라, 그 키 입력으로 40초를 이어 생성한 `compare.mp4` 를 만듭니다.
가로 순서: **실제 게임 | 3스텝 | 5스텝 | 7스텝 | 10스텝** (모두 과거 프레임 노이즈 0.1). 품질이 유지되는 가장 적은 스텝 수를 고르면 됩니다. 다른 설정은 `--settings 5:0.05 5:0.2` 처럼 `스텝:노이즈` 로 지정.

## 5. 학습된 모델 실행 (로컬 PC)

서버의 `runs/wm/last.pt` 를 받아 `checkpoints/last.pt` 에 두고:

```bash
python play_model.py
```

- 실제 게임으로 첫 4프레임을 만든 뒤, 그다음부터는 **모든 화면을 모델이 생성**합니다. 조작은 게임과 같고 `R` 은 새 시드로 다시 시작.
- 기본 설정: 디노이징 5스텝, 과거 프레임 노이즈 0.1. RTX GPU면 16비트 가속 자동 적용.
- 설정 바꾸기: `python play_model.py --steps 10 --ctx-sigma 0` (스텝을 늘리면 선명하지만 느림, 노이즈 0.1은 긴 플레이에서 몬스터·구조물 유지에 도움)
- 창 제목에 생성 fps 표시. 속도가 이상하면 `python tools/bench_infer.py --ckpt checkpoints/last.pt` 로 설정별 속도 측정.

## 모델 요약

DIAMOND 방식의 EDM 확산 U-Net, 픽셀 공간에서 직접 생성 (약 3,200만 파라미터).

- 입력: 노이즈 섞인 다음 프레임 + 직전 4프레임 (15채널, 320×240)
- 조건: 노이즈 세기 + 직전 4틱의 키 입력 → 모든 블록의 정규화 층에 scale/shift로 주입
- U-Net: 320×240 → 160×120 → 80×60 → 40×30(어텐션) → 20×15(어텐션)
- 학습 시 과거 프레임에 노이즈를 섞어 자기 출력을 이어 받을 때의 붕괴를 완화
- 추론: Karras 스케줄 Euler 3스텝

한계: 과거 4프레임(약 0.27초)만 보므로, 화면 밖으로 나간 것은 기억하지 못합니다.

## 에셋 출처

게임의 모든 3D 모델은 **[Kenney](https://kenney.nl)** 의 무료 에셋입니다. 라이선스는 **CC0 (퍼블릭 도메인)** 이라 상업적 이용·수정·재배포가 자유롭고 출처 표기 의무도 없지만, 감사의 뜻으로 밝혀 둡니다.

| 에셋 팩 | 링크 | 게임에서 쓴 것 |
|---|---|---|
| **Mini Dungeon** | https://kenney.nl/assets/mini-dungeon | 플레이어(`character-human`), 검, 포션, 바닥·벽·기둥·바위·통·상자·항아리·테이블·의자·목재 구조물 |
| **Graveyard Kit** | https://kenney.nl/assets/graveyard-kit | 적 캐릭터: 스켈레톤(`character-skeleton`), 좀비(`character-zombie`) |

저장소에는 에셋 파일이 들어 있지 않습니다. `setup_assets.py` 가 위 링크에서 내려받은 뒤, Panda3D에서 애니메이션이 동작하도록 다음처럼 변환합니다.

- **Mini Dungeon 캐릭터**: 몸통과 머리가 같은 뼈대를 쓰는 스킨 2개로 나뉘어 있어 panda3d-gltf 로더가 실패합니다. 스킨 1개로 합칩니다.
- **Graveyard Kit 캐릭터**: 뼈대(스킨) 없이 부위 노드를 직접 움직이는 방식이라 Actor 애니메이션이 동작하지 않습니다. 각 부위를 뼈대에 100% 가중치로 묶은 스킨 메시로 변환합니다.

그 밖에 사용한 라이브러리: [Panda3D](https://www.panda3d.org/) (렌더링), [panda3d-gltf](https://github.com/Moguri/panda3d-gltf) (glTF 로딩), [panda3d-simplepbr](https://github.com/Moguri/panda3d-simplepbr) (셰이딩), [pygame](https://www.pygame.org/) (화면 표시·키 입력), [PyTorch](https://pytorch.org/) (모델).
