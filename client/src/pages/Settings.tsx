
import { motion } from 'framer-motion';
import { Layout } from '@/components/Layout';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { toast } from '@/hooks/use-toast';
import { useAppStore } from '@/store';
import { 
  Monitor,
  Moon,
  Sun,
  Settings as SettingsIcon,
  Bell,
  Shield,
  Trash2,
  Download,
  Palette,
  Zap
} from 'lucide-react';

const Settings = () => {
  const { theme, setTheme } = useAppStore();

  const handleExportData = () => {
    toast({
      title: "Export Started",
      description: "Your data is being prepared for download"
    });
  };

  const handleClearData = () => {
    toast({
      title: "Data Cleared",
      description: "All local data has been cleared",
      variant: "destructive"
    });
  };

  const themeOptions = [
    { value: 'light', label: 'Light', icon: Sun },
    { value: 'dark', label: 'Dark', icon: Moon },
    { value: 'system', label: 'System', icon: Monitor }
  ];

  return (
    <Layout>
      <div className="max-w-4xl mx-auto px-4 lg:px-8 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold mb-2">Settings</h1>
          <p className="text-muted-foreground">
            Customize your StreamlyAI experience
          </p>
        </motion.div>

        <div className="space-y-8">
          {/* Appearance */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <Card className="p-6">
              <div className="flex items-center space-x-2 mb-6">
                <Palette className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold">Appearance</h3>
              </div>

              <div className="space-y-6">
                <div className="space-y-3">
                  <Label>Theme</Label>
                  <Select value={theme} onValueChange={(value: any) => setTheme(value)}>
                    <SelectTrigger className="w-48">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {themeOptions.map((option) => {
                        const Icon = option.icon;
                        return (
                          <SelectItem key={option.value} value={option.value}>
                            <div className="flex items-center space-x-2">
                              <Icon className="h-4 w-4" />
                              <span>{option.label}</span>
                            </div>
                          </SelectItem>
                        );
                      })}
                    </SelectContent>
                  </Select>
                  <p className="text-sm text-muted-foreground">
                    Choose your preferred color scheme
                  </p>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <Label>Reduced Motion</Label>
                    <p className="text-sm text-muted-foreground">
                      Minimize animations and transitions
                    </p>
                  </div>
                  <Switch />
                </div>
              </div>
            </Card>
          </motion.div>

          {/* Performance */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <Card className="p-6">
              <div className="flex items-center space-x-2 mb-6">
                <Zap className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold">Performance</h3>
              </div>

              <div className="space-y-6">
                <div className="space-y-3">
                  <Label>Animation Speed</Label>
                  <Select defaultValue="normal">
                    <SelectTrigger className="w-48">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="slow">Slow</SelectItem>
                      <SelectItem value="normal">Normal</SelectItem>
                      <SelectItem value="fast">Fast</SelectItem>
                      <SelectItem value="none">None</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-sm text-muted-foreground">
                    Control the speed of animations throughout the app
                  </p>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <Label>Auto-save Progress</Label>
                    <p className="text-sm text-muted-foreground">
                      Automatically save your work as you go
                    </p>
                  </div>
                  <Switch defaultChecked />
                </div>
              </div>
            </Card>
          </motion.div>

          {/* Notifications */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <Card className="p-6">
              <div className="flex items-center space-x-2 mb-6">
                <Bell className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold">Notifications</h3>
              </div>

              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <Label>Transcription Complete</Label>
                    <p className="text-sm text-muted-foreground">
                      Get notified when your videos are ready
                    </p>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <Label>Export Complete</Label>
                    <p className="text-sm text-muted-foreground">
                      Get notified when your highlight reels are ready
                    </p>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <Label>Browser Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Receive notifications even when the tab is inactive
                    </p>
                  </div>
                  <Switch />
                </div>
              </div>
            </Card>
          </motion.div>

          {/* Privacy & Data */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <Card className="p-6">
              <div className="flex items-center space-x-2 mb-6">
                <Shield className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold">Privacy & Data</h3>
              </div>

              <div className="space-y-6">
                <div className="space-y-3">
                  <Label>Data Retention</Label>
                  <Select defaultValue="30days">
                    <SelectTrigger className="w-48">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="7days">7 days</SelectItem>
                      <SelectItem value="30days">30 days</SelectItem>
                      <SelectItem value="90days">90 days</SelectItem>
                      <SelectItem value="forever">Forever</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-sm text-muted-foreground">
                    How long to keep your transcripts and data
                  </p>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <Label>Analytics</Label>
                    <p className="text-sm text-muted-foreground">
                      Help improve StreamlyAI by sharing usage data
                    </p>
                  </div>
                  <Switch defaultChecked />
                </div>

                <Separator />

                <div className="flex flex-col sm:flex-row gap-4">
                  <Button
                    variant="outline"
                    onClick={handleExportData}
                    className="flex items-center space-x-2"
                  >
                    <Download className="h-4 w-4" />
                    <span>Export Data</span>
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={handleClearData}
                    className="flex items-center space-x-2"
                  >
                    <Trash2 className="h-4 w-4" />
                    <span>Clear All Data</span>
                  </Button>
                </div>
                <p className="text-sm text-muted-foreground">
                  Export your data or permanently delete all stored information
                </p>
              </div>
            </Card>
          </motion.div>

          {/* About */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
          >
            <Card className="p-6">
              <div className="flex items-center space-x-2 mb-6">
                <SettingsIcon className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold">About</h3>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <Label className="text-muted-foreground">Version</Label>
                    <p className="font-medium">1.0.0</p>
                  </div>
                  <div>
                    <Label className="text-muted-foreground">Build</Label>
                    <p className="font-medium">2025.06.27</p>
                  </div>
                </div>
                
                <Separator />
                
                <p className="text-sm text-muted-foreground">
                  StreamlyAI transforms YouTube videos into searchable transcripts and shareable highlight reels using advanced AI technology.
                </p>
              </div>
            </Card>
          </motion.div>
        </div>
      </div>
    </Layout>
  );
};

export default Settings;
