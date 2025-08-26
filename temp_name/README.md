# Alembic Migration System Setup (TEMP)

This folder contains the initial Alembic migration system setup for the analytics database schema. 

## Structure
- `alembic.ini`: Alembic configuration file (edit `sqlalchemy.url` as needed)
- `migrations/`: Alembic migration scripts and environment
    - `env.py`: Alembic environment script (edit `target_metadata` for autogenerate)
    - `versions/`: Migration scripts (empty, add with `alembic revision`)

## Usage
1. Set your database URL in `alembic.ini` or via environment variable.
2. Run Alembic commands from the project root:
   ```bash
   alembic -c temp_name/alembic.ini upgrade head
   alembic -c temp_name/alembic.ini revision --autogenerate -m "Initial migration"
   ```
3. Edit `env.py` to import your SQLAlchemy Base metadata for autogeneration.

## Next Steps
- Integrate with your analytics models (see `env.py` comment).
- Move/rename this folder as appropriate for your project structure.
- Add migration scripts as your schema evolves.
