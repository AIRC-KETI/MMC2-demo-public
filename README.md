# MMC2-demo

## Run this application using docker compose

1. set environment variables in `.env` files. You can find a sample file(`sample.env`) in the root directory.
```bash
OPENAI_API_KEY="abcd"
```

2. run the application using `docker compose`
```bash
docker compose up
```

## Stop this application (stop all container)

`Ctrl` + `C`


## Down this application (remove all container)
```bash
docker compose down
```


## Acknowledgement

본 연구는 정부(과학기술정보통신부)의 재원으로 지원을 받아 수행된 연구입니다. (No. RS-2022-II220320, 상황인지 및 사용자 이해를 통한 인공지능 기반 1:1 복합대화 기술 개발)