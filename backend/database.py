from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

def get_db_session():
    load_dotenv()

    user_name = os.getenv('MYSQL_USER')
    password = os.getenv('MYSQL_PASSWORD')
    host = os.getenv('MYSQL_HOST')
    port = os.getenv('MYSQL_PORT')
    database = os.getenv('MYSQL_DATABASE')

    db_url = f"mysql+pymysql://{user_name}:{password}@{host}:{port}/{database}"

    engine = create_engine(db_url, echo=True)

    Session_Local =  sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return Session_Local
    

    #  docker exec -it mysql_doccontainer mysql -u user -p
    