# Use an official Node.js image for building the React app
FROM node:18-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json before running npm install (caching optimization)
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy the rest of the app's source code
COPY . .

# Create a .env file with default values
RUN echo "VITE_API_URL=http://localhost:5000" > .env

# Build the app
RUN npm run build

# Use a lightweight nginx image to serve the built app
FROM nginx:alpine

# Copy the built app from the previous stage
COPY --from=0 /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]