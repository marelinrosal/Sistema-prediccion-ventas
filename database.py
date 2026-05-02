"""
Capa de base de datos con SQLAlchemy.
Modelos: Archivo, Venta
Clase:   Database
"""

from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, DateTime, ForeignKey, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import pandas as pd

Base = declarative_base()


#MODELOS

class Archivo(Base):
    __tablename__ = "archivos"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    nombre          = Column(String(255), nullable=False)
    ruta_original   = Column(String(512))
    fecha_carga     = Column(DateTime, default=datetime.now)
    total_registros = Column(Integer)
    ventas          = relationship(
        "Venta", back_populates="archivo", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Archivo id={self.id} nombre={self.nombre}>"


class Venta(Base):
    __tablename__ = "ventas"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    archivo_id      = Column(Integer, ForeignKey("archivos.id"), nullable=False)
    fecha           = Column(DateTime)
    producto        = Column(String(255))
    cantidad        = Column(Float)
    precio_unitario = Column(Float)
    ingresos        = Column(Float)
    categoria       = Column(String(255), nullable=True)
    region          = Column(String(255), nullable=True)
    archivo         = relationship("Archivo", back_populates="ventas")

    def __repr__(self):
        return f"<Venta id={self.id} producto={self.producto} ingresos={self.ingresos}>"


# DATABASE

class Database:
    """Gestiona todas las operaciones con SQLAlchemy sobre SQLite."""

    def __init__(self, db_path: str = "ventas.db"):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    # Guardar DATAFRAME
    def guardar_csv(self, df: pd.DataFrame, nombre: str, ruta: str) -> int:
        """
        Inserta el DataFrame en ventas vinculado a un registro en archivos.
        Devuelve el id del archivo guardado.
        """
        archivo = Archivo(
            nombre=nombre,
            ruta_original=ruta,
            fecha_carga=datetime.now(),
            total_registros=len(df),
        )
        self.session.add(archivo)
        self.session.flush()

        filas = []
        for _, row in df.iterrows():
            filas.append(Venta(
                archivo_id=archivo.id,
                fecha=row["fecha"].to_pydatetime()
                      if hasattr(row["fecha"], "to_pydatetime")
                      else row["fecha"],
                producto=str(row.get("producto", "")),
                cantidad=float(row.get("cantidad", 0)),
                precio_unitario=float(row.get("precio_unitario", 0)),
                ingresos=float(row.get("ingresos", 0)),
                categoria=str(row["categoria"])
                          if "categoria" in row and pd.notna(row["categoria"])
                          else None,
                region=str(row["region"])
                       if "region" in row and pd.notna(row["region"])
                       else None,
            ))

        self.session.bulk_save_objects(filas)
        self.session.commit()
        return archivo.id

    # CARAGAR ARCHIVO COMO DATAFRAME─
    def cargar_archivo(self, archivo_id: int) -> pd.DataFrame:
        ventas = (
            self.session.query(Venta)
            .filter(Venta.archivo_id == archivo_id)
            .all()
        )
        if not ventas:
            return pd.DataFrame()

        rows = [{
            "fecha":           v.fecha,
            "producto":        v.producto,
            "cantidad":        v.cantidad,
            "precio_unitario": v.precio_unitario,
            "ingresos":        v.ingresos,
            "categoria":       v.categoria,
            "region":          v.region,
        } for v in ventas]

        df = pd.DataFrame(rows)
        df["fecha"] = pd.to_datetime(df["fecha"])
        return df

    # LISTAR ARCHIVOS GRANDES
    def listar_archivos(self) -> list[Archivo]:
        return (
            self.session.query(Archivo)
            .order_by(Archivo.fecha_carga.desc())
            .all()
        )

    # ELIMINAR ARCHIVO Y SUS VENTAS
    def eliminar_archivo(self, archivo_id: int) -> bool:
        obj = (
            self.session.query(Archivo)
            .filter(Archivo.id == archivo_id)
            .first()
        )
        if obj:
            self.session.delete(obj)
            self.session.commit()
            return True
        return False

    # ESTADISTICAS
    def stats_archivo(self, archivo_id: int) -> dict:
        result = self.session.execute(text("""
            SELECT
                COUNT(*)                 AS registros,
                SUM(ingresos)            AS total_ingresos,
                AVG(ingresos)            AS promedio_ingreso,
                COUNT(DISTINCT producto) AS productos,
                MIN(fecha)               AS fecha_min,
                MAX(fecha)               AS fecha_max
            FROM ventas
            WHERE archivo_id = :aid
        """), {"aid": archivo_id}).fetchone()

        return {
            "registros":        result[0] or 0,
            "total_ingresos":   result[1] or 0,
            "promedio_ingreso":  result[2] or 0,
            "productos":        result[3] or 0,
            "fecha_min":        result[4],
            "fecha_max":        result[5],
        }

    # CERRAR SESIÓN
    def cerrar(self):
        self.session.close()