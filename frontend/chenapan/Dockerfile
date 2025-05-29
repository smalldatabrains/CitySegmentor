# Build stage
FROM node:18-alpine AS builder

# Install TypeScript globally
RUN npm install -g typescript

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json before running npm install (caching optimization)
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy the rest of the app's source code
COPY . .

# Set permissions for the entire /app directory
RUN chmod -R 755 /app && \
    chown -R node:node /app && \
    chmod +x node_modules/.bin/tsc && \
    chmod +x node_modules/.bin/vite

# Create a .env file with default values
RUN echo "VITE_API_URL=http://localhost:5000" > .env

# Build the app with proper permissions
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy the built app from the previous stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
