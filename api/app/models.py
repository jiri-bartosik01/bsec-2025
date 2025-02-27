from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import (
    DOUBLE_PRECISION,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.services.database import Base

# objectid_1,geometry,zuj,alkohol_vinik,hlavni_pricina,srazka,nasledky,pricina,stav_vozovky,povetrnostni_podm,rozhled,misto_nehody,druh_komun,druh_vozidla,mestska_cast,pohlavi,alkohol,den_v_tydnu,mesic_t,katastr,nasledek,ozn_osoba,zavineni,viditelnost,situovani,osoba,stav_ridic,doba,lz,den,vek,rok_nar,rok,tz,smrt,lehce_zran_os,tezce_zran_os,usmrceno_os,id_vozidla,hodina,ovlivneni_ridice,cas,mesic,e,d,id_nehody,datum,hmotna_skoda_1,skoda_vozidlo,globalid,cluster_id,centroid,radius,id,feature_id,geometry.1,datum_exportu,st_x,st_y,distance,rn,datum.1,teplota_průměrná,teplota_maximální,teplota_minimální,rychlost_větru_,tlak_vzduchu,vlhkost_vzduchu,úhrn_srážek,celková_výška_sněhu,sluneční_svit,cars,trucks


class Point(Base):
    __tablename__ = "dopravni_nehody"

    objectid_1: Mapped[int] = mapped_column(Integer, primary_key=True)
    geometry: Mapped[WKBElement] = mapped_column(Geometry(geometry_type="POINT", srid=4326))
    zuj: Mapped[str] = mapped_column(String)
    alkohol_vinik: Mapped[str] = mapped_column(String)
    hlavni_pricina: Mapped[str] = mapped_column(String)
    srazka: Mapped[str] = mapped_column(String)
    nasledky: Mapped[str] = mapped_column(String)
    pricina: Mapped[str] = mapped_column(String)
    stav_vozovky: Mapped[str] = mapped_column(String)
    povetrnostni_podm: Mapped[str] = mapped_column(String)
    rozhled: Mapped[str] = mapped_column(String)
    misto_nehody: Mapped[str] = mapped_column(String)
    druh_komun: Mapped[str] = mapped_column(String)
    druh_vozidla: Mapped[str] = mapped_column(String)
    mestska_cast: Mapped[str] = mapped_column(String)
    pohlavi: Mapped[str] = mapped_column(String)
    alkohol: Mapped[str] = mapped_column(String)
    den_v_tydnu: Mapped[str] = mapped_column(String)
    mesic_t: Mapped[str] = mapped_column(String)
    katastr: Mapped[str] = mapped_column(String)
    nasledek: Mapped[str] = mapped_column(String)
    ozn_osoba: Mapped[str] = mapped_column(String)
    zavineni: Mapped[str] = mapped_column(String)
    viditelnost: Mapped[str] = mapped_column(String)
    situovani: Mapped[str] = mapped_column(String)
    osoba: Mapped[str] = mapped_column(String)
    stav_ridic: Mapped[str] = mapped_column(String)
    doba: Mapped[str] = mapped_column(String)
    lz: Mapped[str] = mapped_column(String)
    den: Mapped[str] = mapped_column(String)
    vek: Mapped[str] = mapped_column(String)
    rok_nar: Mapped[str] = mapped_column(String)
    rok: Mapped[str] = mapped_column(String)
    tz: Mapped[str] = mapped_column(String)
    smrt: Mapped[str] = mapped_column(String)
    lehce_zran_os: Mapped[str] = mapped_column(String)
    tezce_zran_os: Mapped[str] = mapped_column(String)
    usmrceno_os: Mapped[str] = mapped_column(String)
    id_vozidla: Mapped[str] = mapped_column(String)
    hodina: Mapped[str] = mapped_column(String)
    ovlivneni_ridice: Mapped[str] = mapped_column(String)
    cas: Mapped[str] = mapped_column(String)
    mesic: Mapped[str] = mapped_column(String)
    e: Mapped[str] = mapped_column(String)
    d: Mapped[str] = mapped_column(String)
    id_nehody: Mapped[str] = mapped_column(String)
    datum: Mapped[str] = mapped_column(String)
    hmotna_skoda_1: Mapped[int] = mapped_column(Integer)
    skoda_vozidlo: Mapped[int] = mapped_column(Integer)
    globalid: Mapped[str] = mapped_column(String)
    cluster_id: Mapped[int] = mapped_column(Integer)
    centroid: Mapped[WKBElement] = mapped_column(Geometry(geometry_type="POINT", srid=4326))
    radius: Mapped[float] = mapped_column(DOUBLE_PRECISION)
