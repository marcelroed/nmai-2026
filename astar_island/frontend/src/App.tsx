import { RouterProvider } from "@tanstack/react-router";
import { AppDataProvider } from "./data";
import { router } from "./router";

export default function App() {
  return (
    <AppDataProvider>
      <RouterProvider router={router} />
    </AppDataProvider>
  );
}
