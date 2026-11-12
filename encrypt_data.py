"""Encrypt analysis.json -> data/analysis.enc.  Usage: python encrypt_data.py path/to/analysis.json  (password from env DASH_PASSWORD)"""
import base64, os, sys
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

salt = os.urandom(16)
key = base64.urlsafe_b64encode(Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(os.environ["DASH_PASSWORD"].encode()))
os.makedirs("data", exist_ok=True)
open("data/analysis.enc", "wb").write(salt + Fernet(key).encrypt(open(sys.argv[1], "rb").read()))
print("wrote data/analysis.enc")
