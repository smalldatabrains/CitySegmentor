FROM node:18-alpine

WORKDIR /app

COPY chenapan/package*.json ./

RUN npm install

COPY chenapan/ .

EXPOSE 3001

CMD ["npm", "start"]
