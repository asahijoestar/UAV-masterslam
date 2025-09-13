#!/usr/bin/env python3
# setuzoku.py
import socket, time, subprocess, sys, re

TELLO = ("192.168.10.1", 8889)

def get_local_ip():
    # 192.168.10.x を持つIFのIPを ip コマンドから取得
    out = subprocess.check_output(
        "ip -o -4 addr show | awk '{print $2,$4}'", shell=True
    ).decode()
    for line in out.splitlines():
        if "192.168.10." in line:
            return line.split()[1].split('/')[0]
    return None

def ask(sock, cmd, tries=3, wait=0.3, timeout=2.0):
    sock.settimeout(timeout)
    for i in range(1, tries+1):
        sock.sendto(cmd.encode(), TELLO)
        try:
            data,_ = sock.recvfrom(1024)
            print(f"{cmd} => {data!r}")
            return data.decode(errors="ignore").strip()
        except socket.timeout:
            print(f"{cmd} => (timeout {i}/{tries})")
            time.sleep(wait)
    return None

def main():
    local_ip = get_local_ip()
    if not local_ip:
        print("[NG] 192.168.10.x のIPが見つかりません。TELLO-xxxxxxへ接続してから再実行してください。")
        sys.exit(1)
    print(f"[INFO] bind {local_ip}:9000")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((local_ip, 9000))

    # SDKモードに入る → 少し待つ
    r = ask(s, "command", tries=3, wait=0.5, timeout=2.0)
    time.sleep(0.2)

    # バッテリーとストリームON
    ask(s, "battery?", tries=2, timeout=2.0)
    ask(s, "streamon", tries=2, timeout=2.0)

    s.close()

if __name__ == "__main__":
    main()

