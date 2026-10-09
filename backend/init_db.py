"""Create database tables and, optionally, a user.

    python -m backend.init_db
    python -m backend.init_db --create-user dr.smith --role clinician --full-name "Dr Smith"
"""

import argparse
import getpass

from backend.database import Base, SessionLocal, engine
from backend.models import User
from backend.security import hash_password


def init_database():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


def create_user(username: str, role: str, full_name: str, password: str) -> None:
    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == username).first():
            raise SystemExit(f"User '{username}' already exists.")
        db.add(User(username=username, role=role, full_name=full_name, password_hash=hash_password(password)))
        db.commit()
        print(f"Created {role} '{username}'.")
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--create-user")
    ap.add_argument("--role", choices=["clinician", "admin"], default="clinician")
    ap.add_argument("--full-name", default="")
    args = ap.parse_args()

    init_database()
    if args.create_user:
        password = getpass.getpass("Password (min 8 characters): ")
        if len(password) < 8:
            raise SystemExit("Password must be at least 8 characters.")
        create_user(args.create_user, args.role, args.full_name, password)
