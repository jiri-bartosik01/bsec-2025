FROM postgis/postgis:17-3.5-alpine

ENV POSTGRES_DB=postgres
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

COPY ./db/init.sql /docker-entrypoint-initdb.d/init.sql

EXPOSE 5432

RUN apk add make git clang19 && git clone git@github.com:cybertec-postgresql/kmeans-postgresql.git && cd kmeans-postgresql && make install

